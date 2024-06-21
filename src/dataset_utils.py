import tensorflow as tf
import shutil
import os
import cv2
import numpy as np
import pandas as pd
import random
import augmentation_layer
from sklearn.model_selection import train_test_split
from mtcnn import MTCNN

# detector = MTCNN()

# # Define a function to detect and crop faces using MTCNN face detector
# def detect_and_crop_face(image_path, save_path):
#     img = cv2.imread(image_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Detect faces in the image
#     detections = detector.detect_faces(img_rgb, )

#     if detections:
#         # Assume the largest face is the main face
#         largest_face = max(detections, key=lambda det: det['box'][2] * det['box'][3])
#         x, y, w, h = largest_face['box']
#         cropped_img = img[y:y+h, x:x+w]
#         cv2.imwrite(save_path, cropped_img)
#     else:
#         # If no face is detected, save the original image
#         cv2.imwrite(save_path, img)

def preprocess_np_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp

def preprocess_tf_input(x, data_format=None, version=1):
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        constants = tf.constant([93.5940, 104.7624, 129.1863], dtype=tf.float32)
    elif version == 2:
        constants = tf.constant([91.4953, 103.8827, 131.0912], dtype=tf.float32)
    else:
        raise NotImplementedError

    if data_format == 'channels_first':
        x = tf.reverse(x, axis=[2])
    else:
        x = tf.reverse(x, axis=[-1])

    x = tf.cast(x, tf.float32)
    x = tf.subtract(x, constants)

    return x

# Prepare Triplet Dataset for TensorFlow
def prepare_triplet_dataset(triplets, batch_size, shuffle=False):
    def preprocess_triplet(triplet):
        anchor = load_and_preprocess_image(triplet[0])
        pos = load_and_preprocess_image(triplet[1])
        neg = load_and_preprocess_image(triplet[2])
        # Filter out None values
        if anchor is None or pos is None or neg is None:
            return None
        return tf.stack([anchor, pos, neg], axis=0)
    
    dataset = tf.data.Dataset.from_tensor_slices(triplets)
    dataset = dataset.map(preprocess_triplet,
        num_parallel_calls=tf.data.AUTOTUNE
    ).filter(lambda x: x is not None)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(triplets))

    dataset = dataset.batch(batch_size)
    return dataset

def generate_triplets(image_dir='face_recognition/datasets/celeb_a/img_align_celeba_cropped',
                      identity_file='face_recognition/datasets/celeb_a/identity_CelebA.txt'):
    # Dictionary to store images for each identity
    identity_images = {}
    
    # Read identity file and populate identity_images dictionary
    with open(identity_file, 'r') as file:
        for line in file:
            image, identity = line.strip().split()
            if identity not in identity_images:
                identity_images[identity] = []
            identity_images[identity].append(image)
    
    # Filter identities with less than 3 images
    valid_identities = [identity for identity, images in identity_images.items() if len(images) >= 3]
    
    # Initialize lists for triplets
    triplets = []
    
    # Generate triplets
    for identity in valid_identities:
        images = identity_images[identity]
        for anchor in images:
            # Positive image can be any other image from the same identity
            positive = random.choice(images)
            while positive == anchor:
                positive = random.choice(images)
            
            # Select a different identity for negative image
            negative_identity = random.choice(valid_identities)
            while negative_identity == identity:
                negative_identity = random.choice(valid_identities)
            
            # Select a random image from the negative identity
            negative = random.choice(identity_images[negative_identity])
            
            # Ensure no None values in the triplet
            if anchor is not None and positive is not None and negative is not None:
                triplets.append((anchor, positive, negative))
            else:
                print("Warning: None value found in triplet")
    
    return triplets

def split_triplets(triplets, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Shuffle triplets
    random.shuffle(triplets)
    
    # Calculate number of triplets for each set
    num_triplets = len(triplets)
    num_train = int(train_ratio * num_triplets)
    num_val = int(val_ratio * num_triplets)
    num_test = num_triplets - num_train - num_val
    
    # Split triplets into training, validation, and test sets
    train_triplets = triplets[:num_train]
    val_triplets = triplets[num_train:num_train+num_val]
    test_triplets = triplets[num_train+num_val:]
    
    return train_triplets, val_triplets, test_triplets

def load_and_preprocess_image(file_path):
    try:
        # Load image
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        # Resize to VGG16 input shape
        image = tf.image.resize(image, (224, 224))
        # Normalise pixel values to VGGFace2 dataset
        # image = preprocess_tf_input(image)
        return image
    except Exception as e:
        # Handle error (e.g., log it)
        print(f"Error loading or preprocessing image: {e}")
        return None

# Prepare CelebA Dataset for Training
def prepare_celeba_fr(
    base_img_dir,
    identity_file='face_recognition/datasets/celeb_a/identity_CelebA.txt',
    base_split_dir='face_recognition/datasets/celeb_a/split_fr_cropped',
    loss_func='categorical_crossentropy',
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
):
    # Load identity file
    identity_df = pd.read_csv(identity_file, sep=' ', header=None, names=['filename', 'identity'])
    identity_df['identity'] = identity_df['identity'].astype(int)

    # Sort identities to ensure consistent mapping
    unique_ids = sorted(identity_df['identity'].unique())

    # Create a label mapping dictionary (identity -> integer index)
    id_to_label = {identity: (label + 1) for label, identity in enumerate(unique_ids)}  # Labels start from 1
    # Add a new column for the mapped integer labels
    identity_df['label'] = identity_df['identity'].map(id_to_label)

    # Total number of classes/identities
    num_ids = len(unique_ids)

    # Check label shape
    assert identity_df['label'].shape[0] == identity_df.shape[0], "Label shape mismatch"
    # Check label assignment
    for index, row in identity_df.iterrows():
        assert id_to_label[row['identity']] == row['label'], f"Label assignment error at index {index}"

    # Preprocess dataset directory
    if os.path.exists(base_split_dir):
        print("Split directories already exist. Skipping splitting process.")
    else:
        # Create split directories
        for partition_name in ['train', 'val', 'test']:
            partition_dir = os.path.join(base_split_dir, partition_name)
            if not os.path.exists(partition_dir):
                os.makedirs(partition_dir)
            for identity in unique_ids:
                identity_dir = os.path.join(partition_dir, str(identity))
                if not os.path.exists(identity_dir):
                    os.makedirs(identity_dir)

        for identity in unique_ids:
            identity_images = identity_df[identity_df['identity'] == identity]['filename'].tolist()

            # Edge case handling
            if len(identity_images) <= 1:
                # If there is only one sample, assign it to both train and test sets
                train_images = identity_images
                test_images = identity_images
                val_images = []
            elif len(identity_images) <= 2:
                # If there are only two samples, assign one to train and one to test, leave validation empty
                train_images = identity_images[:1]
                test_images = identity_images[1:]
                val_images = []
            else:
                train_images, temp_images = train_test_split(identity_images, train_size=train_ratio, random_state=42)
            if len(temp_images) <= 1:
                # If there is only one sample left for validation, assign it to validation and leave test empty
                val_images = temp_images
                test_images = []
            else:
                val_images, test_images = train_test_split(temp_images, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)

            # Move images to their respective directories
            for img in train_images:
                src_path = os.path.join(base_img_dir, img)
                dest_path = os.path.join(base_split_dir, 'train', str(identity), img)
                # detect_and_crop_face(src_path, dest_path)
                shutil.copy(src_path, dest_path)

            for img in val_images:
                src_path = os.path.join(base_img_dir, img)
                dest_path = os.path.join(base_split_dir, 'val', str(identity), img)
                # detect_and_crop_face(src_path, dest_path)
                shutil.copy(src_path, dest_path)
            
            for img in test_images:
                src_path = os.path.join(base_img_dir, img)
                dest_path = os.path.join(base_split_dir, 'test', str(identity), img)
                # detect_and_crop_face(src_path, dest_path)
                shutil.copy(src_path, dest_path)

    # Instantiate synthetic augmentation layer
    synthetic_augmentation = augmentation_layer.RandomOcclusionLayer(
        augmentation_prob=0.3,  # with 40% chance of synthetic occlusion
        sunglasses_path='face_recognition/datasets/augment/black_sunglasses.png',
        hat_path='face_recognition/datasets/augment/hat.png',
        mask_path='face_recognition/datasets/augment/mask.png'
    )
    
    def augment_image(image, label):
        # Apply custom augmentation layer
        image = synthetic_augmentation(image, training=True)
        return image, label
    
    # Uses mapping and SCCE
    def get_image_paths_and_labels(partition):
        image_paths = []
        labels = []
        partition_dir = os.path.join(base_split_dir, partition)
        for identity in unique_ids:
            identity_dir = os.path.join(partition_dir, str(identity))
            for img_name in os.listdir(identity_dir):
                image_paths.append(os.path.join(identity_dir, img_name))
                labels.append(id_to_label[identity])  # or use 'labels.append(identity)'
        return image_paths, labels

    train_image_paths, train_labels = get_image_paths_and_labels('train')
    val_image_paths, val_labels = get_image_paths_and_labels('val')
    test_image_paths, test_labels = get_image_paths_and_labels('test')

    # Check label shape
    assert np.array(train_labels).shape == (len(train_labels),), "Train labels shape mismatch"
    assert np.array(val_labels).shape == (len(val_labels),), "Validation labels shape mismatch"
    assert np.array(test_labels).shape == (len(test_labels),), "Test labels shape mismatch"
    # Check label range
    assert min(train_labels) >= 1 and max(train_labels) <= num_ids, "Train labels out of range"
    assert min(val_labels) >= 1 and max(val_labels) <= num_ids, "Validation labels out of range"
    assert min(test_labels) >= 1 and max(test_labels) <= num_ids, "Test labels out of range"

    if loss_func == 'sparse_categorical_crossentropy':
        # Convert to tensors
        train_labels = tf.constant(train_labels, dtype=tf.int32)
        val_labels = tf.constant(val_labels, dtype=tf.int32)
        test_labels = tf.constant(test_labels, dtype=tf.int32)
    elif loss_func == 'categorical_crossentropy':
        # One-hot encoding for 'categorical crossentropy' (CCE)
        train_labels = tf.one_hot(train_labels, num_ids)
        val_labels = tf.one_hot(val_labels, num_ids)
        test_labels = tf.one_hot(test_labels, num_ids)
    
    print(f"Train labels shape (sparse): {train_labels.shape}")
    print(f"Validation labels shape (sparse): {val_labels.shape}")
    print(f"Test labels shape (sparse): {test_labels.shape}")

    datasets = {
            'train': tf.data.Dataset.from_tensor_slices((tf.constant(train_image_paths), train_labels))
                                    .map(lambda img_path, label:
                                         (load_and_preprocess_image(img_path), label),
                                         num_parallel_calls=tf.data.AUTOTUNE)
                                    # .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
                                    .batch(batch_size),
            'val': tf.data.Dataset.from_tensor_slices((tf.constant(val_image_paths), val_labels))
                                    .map(lambda img_path, label:
                                         (load_and_preprocess_image(img_path), label),
                                         num_parallel_calls=tf.data.AUTOTUNE)
                                    # .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
                                    .batch(batch_size),
            'test_original': tf.data.Dataset.from_tensor_slices((tf.constant(test_image_paths), test_labels))
                                        .map(lambda img_path, label:
                                             (load_and_preprocess_image(img_path), label),
                                             num_parallel_calls=tf.data.AUTOTUNE)
                                        .batch(batch_size),
            'test_augmented': tf.data.Dataset.from_tensor_slices((tf.constant(test_image_paths), test_labels))
                                        .map(lambda img_path, label:
                                             (load_and_preprocess_image(img_path), label),
                                             num_parallel_calls=tf.data.AUTOTUNE)
                                        # .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
                                        .batch(batch_size)
        }
    
    return datasets['train'], datasets['val'], datasets['test_original'], datasets['test_augmented'], num_ids

def prepare_celeba_extract(
    base_img_dir,
    identity_file='face_recognition/datasets/celeb_a/identity_CelebA.txt',
    partition_file='face_recognition/datasets/celeb_a/list_eval_partition.txt',
    base_split_dir='face_recognition/datasets/celeb_a/split',
    batch_size=32,
    test_split_ratio=0.5
):
    # Load identity and partition information
    identity_df = pd.read_csv(identity_file, sep=' ', header=None, names=['filename', 'identity'])
    partition_df = pd.read_csv(partition_file, sep=' ', header=None, names=['filename', 'partition'])
    df = pd.merge(identity_df, partition_df, on='filename')
    df['identity'] = df['identity'].astype(int)

    # number of classes
    unique_ids = df['identity'].unique()
    num_ids = len(unique_ids)

    partitions = {'train': 0, 'val': 1, 'test': 2}

    def filter_empty_identity_directories(base_split_dir, partitions=['train', 'val', 'test']):
        for partition in partitions:
            partition_dir = os.path.join(base_split_dir, partition)
            for identity_dir in os.listdir(partition_dir):
                if identity_dir == '.DS_Store':
                    continue  # Skip .DS_Store files created by MacOS
                identity_path = os.path.join(partition_dir, identity_dir)

                # Check if the directory is empty
                if not os.listdir(identity_path):
                    # print(f"Removing empty identity directory: {identity_path}")
                    os.rmdir(identity_path)

    # Split the dataset if not already split
    if not os.path.exists(base_split_dir):
        os.makedirs(base_split_dir)
        for partition_name in ['train', 'val', 'test']:
            os.makedirs(os.path.join(base_split_dir, partition_name))
            # Create subdirectories for each identity within the partition
            for identity in unique_ids:
                os.makedirs(os.path.join(base_split_dir, partition_name, str(identity)))

        for _, row in df.iterrows():
            src_path = os.path.join(base_img_dir, row['filename'])
            partition_name = [k for k, v in partitions.items() if v == row['partition']][0]  

            # Move to the specific identity subdirectory
            dst_path = os.path.join(base_split_dir, partition_name, str(row['identity']), row['filename'])

            # shutil.move(src_path, dst_path)
            shutil.copy2(src_path, dst_path)

    # Filter out empty directories in train, val, and test sets
    filter_empty_identity_directories(base_split_dir)

    # Create train/val/test image and label lists
    image_paths = {partition: [] for partition in partitions}
    labels = {partition: [] for partition in partitions}
    for _, row in df.iterrows():
        partition = [k for k, v in partitions.items() if v == row['partition']][0]
        # Construct path based on partition, base_split_dir, and identity
        image_paths[partition].append(os.path.join(base_split_dir, partition, str(row['identity']), row['filename']))
        labels[partition].append(row['identity'])

    # One-hot encoding for labels
    labels = {k: tf.one_hot(v, num_ids) for k, v in labels.items()}

    # Split the test set
    test_image_paths = image_paths['test']
    test_labels = labels['test']
    split_index = int(len(test_image_paths) * test_split_ratio)
    test_image_paths_original, test_image_paths_augmented = test_image_paths[:split_index], test_image_paths[split_index:]
    test_labels_original, test_labels_augmented = test_labels[:split_index], test_labels[split_index:]

    def load_and_preprocess_image(file_path, label):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.vgg16.preprocess_input(image)
        # label = tf.cast(label, tf.int32)
        return image, label

    # Instantiate synthetic augmentation layer
    synthetic_augmentation = augmentation_layer.RandomOcclusionLayer(
        augmentation_prob=0.4,  # with 40% chance of synthetic occlusion (sunglasses, mask, hat)
        sunglasses_path='face_recognition/datasets/augment/black_sunglasses.png',
        hat_path='face_recognition/datasets/augment/hat.png',
        mask_path='face_recognition/datasets/augment/mask.png'
    )

    def augment_image(image, label):
        # Apply custom augmentation layer
        image = synthetic_augmentation(image, training=True)
        return image, label
    
    datasets = {
        'train': tf.data.Dataset.from_tensor_slices((tf.constant(image_paths['train']), labels['train']))
                                .map(load_and_preprocess_image)
                                .map(augment_image)
                                .batch(batch_size),
        'val': tf.data.Dataset.from_tensor_slices((tf.constant(image_paths['val']), labels['val']))
                               .map(load_and_preprocess_image)
                               .batch(batch_size),
        'test_original': tf.data.Dataset.from_tensor_slices((tf.constant(test_image_paths_original), test_labels_original))
                                      .map(load_and_preprocess_image)
                                      .batch(batch_size),
        'test_augmented': tf.data.Dataset.from_tensor_slices((tf.constant(test_image_paths_augmented), test_labels_augmented))
                                       .map(load_and_preprocess_image)
                                       .map(augment_image)
                                       .batch(batch_size)
    }

    return datasets['train'], datasets['val'], datasets['test_original'], datasets['test_augmented'], num_ids
