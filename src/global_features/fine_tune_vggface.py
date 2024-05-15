import tensorflow as tf
import os
import pandas as pd
import random
import numpy as np
import shutil

def add_sunglasses_occlusion(image):
    sunglasses_img = tf.io.read_file("path/to/sunglasses_image.jpg") 
    sunglasses_img = tf.image.decode_jpeg(sunglasses_img, channels=3)

    # Resize sunglasses image to match the target image dimensions
    sunglasses_img = tf.image.resize(sunglasses_img, (image.shape[0], image.shape[1])) 

    # Randomly position the sunglasses (adjusted for resized image)
    x = random.randint(0, image.shape[1] - sunglasses_img.shape[1])
    y = random.randint(0, image.shape[0] - sunglasses_img.shape[0])

    # Crop the image to match the sunglasses image shape
    image_cropped = tf.image.crop_to_bounding_box(image, y, x, sunglasses_img.shape[0], sunglasses_img.shape[1])

    # Overlay sunglasses using alpha blending
    alpha = tf.ones_like(sunglasses_img) * 0.8 
    image_with_sunglasses = sunglasses_img * alpha + image_cropped * (1 - alpha)

    # Paste the cropped image back into the original image
    image = tf.image.pad_to_bounding_box(image_with_sunglasses, y, x, image.shape[0], image.shape[1])

    return image

def add_random_rectangle_occlusion(image):
    """Adds a random rectangular occlusion to an image."""

    batch_size = tf.shape(image)[0]  # Get the batch size
    boxes = tf.TensorArray(tf.float32, size=batch_size)

    for i in tf.range(batch_size):
        height = random.randint(30, 80)
        width = random.randint(50, 120)
        x = random.randint(0, image.shape[1] - width)
        y = random.randint(0, image.shape[0] - height)
        boxes = boxes.write(i, [[y / image.shape[0], x / image.shape[1], (y + height) / image.shape[0], (x + width) / image.shape[1]]])

    boxes = boxes.stack()  
    boxes = tf.expand_dims(boxes, 1)  # Add a dimension for num_boxes 

    image = tf.image.draw_bounding_boxes(image, boxes, tf.zeros([batch_size, 1, 3]))  # Black color for the boxes
    return image

def prepare_celeba_dataset(
    base_img_dir,
    identity_file='face_recognition/datasets/celeb_a/identity_CelebA.txt',
    partition_file='face_recognition/datasets/celeb_a/list_eval_partition.txt',
    base_split_dir='face_recognition/datasets/celeb_a/split',
    image_size=(224, 224),
    batch_size=32,
    test_split_ratio=0.5
):
    # Load identity and partition information
    identity_df = pd.read_csv(identity_file, sep=' ', header=None, names=['filename', 'identity'])
    partition_df = pd.read_csv(partition_file, sep=' ', header=None, names=['filename', 'partition'])
    df = pd.merge(identity_df, partition_df, on='filename')
    df['identity'] = df['identity'].astype(int)

    unique_ids = df['identity'].unique()
    num_ids = len(unique_ids)

    partitions = {'train': 0, 'val': 1, 'test': 2}

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

            shutil.move(src_path, dst_path)
            # shutil.copy2(src_path, dst_path)

    # Create train/val/test image and label lists
    image_paths = {partition: [] for partition in partitions}
    labels = {partition: [] for partition in partitions}
    for _, row in df.iterrows():
        partition = [k for k, v in partitions.items() if v == row['partition']][0]
        # Construct path based on partition, base_split_dir, and identity
        image_paths[partition].append(os.path.join(base_split_dir, partition, str(row['identity']), row['filename']))
        labels[partition].append(row['identity'])

    # Split the test set
    test_image_paths = image_paths['test']
    test_labels = labels['test']
    split_index = int(len(test_image_paths) * test_split_ratio)
    test_image_paths_original, test_image_paths_augmented = test_image_paths[:split_index], test_image_paths[split_index:]
    test_labels_original, test_labels_augmented = test_labels[:split_index], test_labels[split_index:]

    # Create tf.data Datasets
    def load_and_preprocess_image(file_path, label):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = tf.keras.applications.vgg16.preprocess_input(image)
        label = tf.cast(label, tf.int32)  # Ensure label is an integer
        label = tf.one_hot(label, num_ids)
        return image, label

    def augment_image(image, label):
        # Geometric Transformations
        if random.random() < 0.5:
            image = tf.image.random_flip_left_right(image)  # Random horizontal flip
        if random.random() < 0.5:
            image = tf.image.rot90(image, k=random.randint(1, 3))  # Random 90-degree rotation

        # Color Transformations
        if random.random() < 0.5:
            image = tf.image.random_brightness(image, max_delta=0.2)  # Random brightness adjustment
        if random.random() < 0.5:
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Random contrast adjustment
        if random.random() < 0.5:
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  # Random saturation adjustment

        # Occlusions (with probability)
        if random.random() < 0.3:  # 30% chance of occlusion
            occlusion_fn = random.choice([add_sunglasses_occlusion, add_random_rectangle_occlusion])
            image = occlusion_fn(image)

        return image, label

    # datasets = {}
    # for partition in partitions:
    #     datasets[partition] = tf.data.Dataset.from_tensor_slices((tf.constant(image_paths[partition]), labels[partition]))
    #     datasets[partition] = datasets[partition].map(load_and_preprocess_image)
    #     if partition == 'train':
    #         datasets[partition] = datasets[partition].map(augment_image)
    #     datasets[partition] = datasets[partition].batch(batch_size)

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

    # return datasets['train'], datasets['val'], datasets['test'], num_ids

# Load CelebA dataset
train_dataset, validation_dataset, test_dataset_original, test_dataset_augmented, num_classes = prepare_celeba_dataset('face_recognition/datasets/celeb_a/img_align_celeba')

# Load VGG16 base model with VGGFace weights (without top)
base_model = tf.keras.applications.VGG16(weights=None,
                                         include_top=False, input_shape=(224, 224, 3))
weights_path = 'face_recognition/src/global_features/rcmalli_vggface_tf_notop_vgg16.h5'
base_model.load_weights(weights_path)

# Freeze base model layers (optional)
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # Get num_classes from dataset

# Create the final model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Performance optimization: prefetch and cache data
train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset_original = test_dataset_original.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset_augmented = test_dataset_augmented.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Fine-tune the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)

# Evaluate on the original test set
original_test_loss, original_test_accuracy = model.evaluate(test_dataset_original)

# Evaluate on the augmented test set
augmented_test_loss, augmented_test_accuracy = model.evaluate(test_dataset_augmented)

print("Original Test Accuracy:", original_test_accuracy)
print("Augmented Test Accuracy:", augmented_test_accuracy)