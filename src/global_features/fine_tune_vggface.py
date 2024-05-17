import tensorflow as tf
import os
import pandas as pd
import random
import numpy as np
import shutil
import augmentation

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
            occlusion_fn = random.choice([augmentation.occlude_rectangle, augmentation.add_sunglasses_to_image])
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

# Save the entire model
model.save('face_recognition_model')  # Creates a directory with model structure and weights

# Save just the weights (smaller file size)
model.save_weights('face_recognition_model_weights.h5')  # Only saves the weights 

# # Option 1: Load entire model
# loaded_model = tf.keras.models.load_model('face_recognition_model')

# # Option 2: Load just weights, then recreate model structure
# loaded_model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)  # Your original model structure
# loaded_model.load_weights('face_recognition_model_weights.h5') 

### EXTRACT GLOBAL FEATURES
# # Create a new model that outputs the penultimate layer
# feature_extractor = tf.keras.Model(inputs=loaded_model.input, outputs=loaded_model.layers[-2].output)  

# # ... (Load your image or dataset) ...
# features = feature_extractor.predict(your_image_data)

# # Load your image or dataset...
# predictions = loaded_model.predict(your_image_data)


##### EXTRACT
# # Load the model
# loaded_model = tf.keras.models.load_model('face_recognition_model')

# # Create feature extractor
# feature_extractor = tf.keras.Model(inputs=loaded_model.input, outputs=loaded_model.layers[-2].output)  

# # Get features for an image
# image_path = 'path/to/your/image.jpg'
# img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
# img_array = tf.keras.preprocessing.image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)  
# img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
# features = feature_extractor.predict(img_array)