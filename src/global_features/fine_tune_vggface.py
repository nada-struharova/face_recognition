import tensorflow as tf
import os
import pandas as pd
import random
import numpy as np

def add_sunglasses_occlusion(image):
    """Adds a realistic sunglasses occlusion to an image."""
    sunglasses_img = tf.io.read_file("path/to/sunglasses_image.jpg")  # Load sunglasses image
    sunglasses_img = tf.image.decode_jpeg(sunglasses_img, channels=3)
    sunglasses_img = tf.image.resize(sunglasses_img, (80, 150))  # Resize as needed

    # Randomly position the sunglasses
    x = random.randint(0, image.shape[1] - 150)
    y = random.randint(0, image.shape[0] - 80)

    # Overlay sunglasses using alpha blending
    alpha = tf.ones_like(sunglasses_img) * 0.8  # 80% opacity
    image = tf.where(
        tf.broadcast_to(alpha > 0, tf.shape(image)),
        sunglasses_img * alpha + image * (1 - alpha),
        image
    )
    return image

# def add_mask_occlusion(image):
#     """Adds a realistic mask occlusion to an image."""
#     # Similar implementation as sunglasses, but with a different image and positioning
#     # Load mask image, resize, randomly position, and alpha blend
#     pass  # Implement this function similarly to the sunglasses one

def add_random_rectangle_occlusion(image):
    """Adds a random rectangular occlusion to an image."""
    height = random.randint(30, 80)
    width = random.randint(50, 120)
    x = random.randint(0, image.shape[1] - width)
    y = random.randint(0, image.shape[0] - height)
    image = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
        tf.constant([[y / image.shape[0], x / image.shape[1], (y + height) / image.shape[0], (x + width) / image.shape[1]]]),
        tf.constant([[0.0, 0.0, 0.0]])  # Black color for the box
    )[0]
    return image

def prepare_celeba_dataset(
    base_img_dir,
    identity_file='face_recognition/datasets/celeb_a/identity_CelebA.txt',
    partition_file='face_recognition/datasets/celeb_a/list_eval_partition.txt',
    base_split_dir='face_recognition/datasets/celeb_a/split',
    image_size=(224, 224),
    batch_size=32
):
    # Load identity and partition information
    identity_df = pd.read_csv(identity_file, sep=' ', header=None, names=['filename', 'identity'])
    partition_df = pd.read_csv(partition_file, sep=' ', header=None, names=['filename', 'partition'])
    df = pd.merge(identity_df, partition_df, on='filename')
    df['identity'] = df['identity'].astype(int)

    unique_ids = df['identity'].unique()
    num_ids = len(unique_ids)

    # Create train/val/test image and label lists (modified)
    partitions = {'train': 0, 'val': 1, 'test': 2}
    image_paths = {partition: [] for partition in partitions}
    labels = {partition: [] for partition in partitions}
    for _, row in df.iterrows():
        partition = [k for k, v in partitions.items() if v == row['partition']][0]
        # Construct path based on partition, base_split_dir, and identity
        image_paths[partition].append(os.path.join(base_split_dir, partition, str(row['identity']), row['filename']))
        labels[partition].append(row['identity'])

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
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        return image, label

    datasets = {}
    for partition in partitions:
        datasets[partition] = tf.data.Dataset.from_tensor_slices((tf.constant(image_paths[partition]), labels[partition]))
        datasets[partition] = datasets[partition].map(load_and_preprocess_image)
        if partition == 'train':
            datasets[partition] = datasets[partition].map(augment_image)
        datasets[partition] = datasets[partition].batch(batch_size)

    return datasets['train'], datasets['val'], datasets['test'], num_ids

# Load CelebA dataset
train_dataset, validation_dataset, test_dataset, num_classes = prepare_celeba_dataset('img_align_celeba')

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
test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Fine-tune the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(test_dataset)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)