import tensorflow as tf
import shutil
import os
import uuid
import numpy as np
import pandas as pd
import augmentation_layer
from sklearn.model_selection import train_test_split

# import tensorflow_datasets as tfds
# import cv2
# import lfw_augmentation

# def load_dataset():
#     (ds, train_ds, val_ds, test_ds), metadata = tfds.load(
#         'lfw',
#         data_dir='face_recognition/datasets/lfw',
#         split=['train', 'train[:80%]', 'train[80%:90%]', 'train[90%:]'],
#         as_supervised=True,
#         with_info=True,
#         batch_size=32
#     )
    
#     train_ds = train_ds.map(
#         lfw_augmentation.add_occlusion, num_parallel_calls=tf.data.AUTOTUNE)
#     val_ds = val_ds.map(
#         lfw_augmentation.add_occlusion, num_parallel_calls=tf.data.AUTOTUNE)
#     test_ds = test_ds.map(
#         lfw_augmentation.add_occlusion, num_parallel_calls=tf.data.AUTOTUNE)

def load_and_preprocess_image(file_path, label, bbox_info=None):
    ## Preprocess CelebA for VGG16
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)

    if bbox_info is not None:
        # Get bounding box info
        bbox = bbox_info[file_path.numpy().decode("utf-8")]
        x, y, width, height = bbox

        # Crop image based on bounding box
        image = tf.image.crop_to_bounding_box(image, y, x, height, width)
        
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.vgg16.preprocess_input(image)
    # label = tf.cast(label, tf.int32)
    return image, label

# Prepare CelebA Dataset for Training
def prepare_celeba_fr(
    base_img_dir,
    identity_file='face_recognition/datasets/celeb_a/identity_CelebA.txt',
    bbox_file='face_recognition/datasets/celeb_a/list_bbox_celeba.txt',
    base_split_dir='face_recognition/datasets/celeb_a/split_fr_bbox',
    loss_func='sparse_categorical_crossentropy',
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
):
    # Load identity file
    identity_df = pd.read_csv(identity_file, sep=' ', header=None, names=['filename', 'identity'])
    identity_df['identity'] = identity_df['identity'].astype(int)

    # Load bounding box file
    bbox_df = pd.read_csv(bbox_file, sep='\s+', skiprows=1, names=['filename', 'x', 'y', 'width', 'height'])
    bbox_info = {row['filename']: [row['x'], row['y'], row['width'], row['height']] for _, row in bbox_df.iterrows()}

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
                shutil.copy(os.path.join(base_img_dir, img), os.path.join(base_split_dir, 'train', str(identity), img))
            
            for img in val_images:
                shutil.copy(os.path.join(base_img_dir, img), os.path.join(base_split_dir, 'val', str(identity), img))
            
            for img in test_images:
                shutil.copy(os.path.join(base_img_dir, img), os.path.join(base_split_dir, 'test', str(identity), img))

    # Instantiate synthetic augmentation layer
    synthetic_augmentation = augmentation_layer.RandomOcclusionLayer(
        augmentation_prob=0.6,  # with 40% chance of synthetic occlusion
        sunglasses_path='face_recognition/datasets/augment/black_sunglasses.png',
        hat_path='face_recognition/datasets/augment/hat.png',
        mask_path='face_recognition/datasets/augment/mask.png'
    )
    
    def augment_image(image, label):
        # Apply custom augmentation layer
        image = synthetic_augmentation(image, training=True)
        return image, label
    
    # def augment_image_train(image, label):
    #     image = tf.identity(image)  # Create a copy of the image
    #     # Apply custom augmentation layer
    #     image = synthetic_augmentation(image, training=True)
    #     return image, label
    
    # # Function to save augmented images
    # def save_augmented_image(image, label):
    #     filename = str(uuid.uuid4()) + ".jpg"
    #     base_split_dir = 'face_recognition/datasets/celeb_a/img_align_celeba'  # Update this to your correct path

    #     def _save_img(img, lbl):
    #         filepath = os.path.join(base_split_dir, 'train', str(lbl.numpy()), filename)
    #         tf.keras.utils.save_img(filepath, img.numpy())  # Convert to NumPy before saving
    #         return img, lbl

    #     return tf.py_function(_save_img, inp=[image, label], Tout=[tf.float32, tf.int32])

    # Uses mapping and SCCE
    def get_image_paths_and_labels(partition):
        image_paths = []
        labels = []
        partition_dir = os.path.join(base_split_dir, partition)
        for identity in unique_ids:  # Iterate over unique identities
            identity_dir = os.path.join(partition_dir, str(identity))
            for img_name in os.listdir(identity_dir):
                image_paths.append(os.path.join(identity_dir, img_name))
                labels.append(id_to_label[identity])  # Use the integer label from the mapping
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

        print("train label: ", train_labels)
    
    print(f"Train labels shape (sparse): {train_labels.shape}")
    print(f"Validation labels shape (sparse): {val_labels.shape}")
    print(f"Test labels shape (sparse): {test_labels.shape}")

    datasets = {
            'train': tf.data.Dataset.from_tensor_slices((tf.constant(train_image_paths), train_labels))
                                    .map(lambda x, y: load_and_preprocess_image(x, y, bbox_info), num_parallel_calls=tf.data.AUTOTUNE)
                                    # .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
                                    # .map(save_augmented_image, num_parallel_calls=tf.data.AUTOTUNE) # Add this line
                                    # .shuffle(buffer_size=len(train_image_paths))
                                    .batch(batch_size),
            'val': tf.data.Dataset.from_tensor_slices((tf.constant(val_image_paths), val_labels))
                                .map(lambda x, y: load_and_preprocess_image(x, y, bbox_info), num_parallel_calls=tf.data.AUTOTUNE)
                                # .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
                                .batch(batch_size),
            'test_original': tf.data.Dataset.from_tensor_slices((tf.constant(test_image_paths), test_labels))
                                        .map(lambda x, y: load_and_preprocess_image(x, y, bbox_info), num_parallel_calls=tf.data.AUTOTUNE)
                                        .batch(batch_size),
            'test_augmented': tf.data.Dataset.from_tensor_slices((tf.constant(test_image_paths), test_labels))
                                        .map(lambda x, y: load_and_preprocess_image(x, y, bbox_info), num_parallel_calls=tf.data.AUTOTUNE)
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
                    continue  # Skip .DS_Store files
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

# Original
# def get_image_paths_and_labels(partition):
#     image_paths = []
#     labels = []
#     partition_dir = os.path.join(base_split_dir, partition)
#     for identity in unique_ids:
#         identity_dir = os.path.join(partition_dir, str(identity))
#         for img_name in os.listdir(identity_dir):
#             image_paths.append(os.path.join(identity_dir, img_name))
#             labels.append(identity)
#     return image_paths, labels
