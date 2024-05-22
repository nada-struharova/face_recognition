import tensorflow as tf
import os
import pandas as pd
import shutil
import augmentation_layer
from sklearn.model_selection import train_test_split

def prepare_celeba_fr(
    base_img_dir,
    identity_file='face_recognition/datasets/celeb_a/identity_CelebA.txt',
    base_split_dir='face_recognition/datasets/celeb_a/split_fr',
    image_size=(224, 224),
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
):
    # Load identity information
    identity_df = pd.read_csv(identity_file, sep=' ', header=None, names=['filename', 'identity'])
    identity_df['identity'] = identity_df['identity'].astype(int)

    unique_ids = identity_df['identity'].unique()
    num_ids = len(unique_ids)

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

        # Split images into train, val, and test
        train_images, temp_images = train_test_split(identity_images, train_size=train_ratio, random_state=42)
        val_images, test_images = train_test_split(temp_images, test_size=test_ratio/(val_ratio + test_ratio), random_state=42)

        # Move images to their respective directories
        for img in train_images:
            shutil.copy(os.path.join(base_img_dir, img), os.path.join(base_split_dir, 'train', str(identity), img))
        
        for img in val_images:
            shutil.copy(os.path.join(base_img_dir, img), os.path.join(base_split_dir, 'val', str(identity), img))
        
        for img in test_images:
            shutil.copy(os.path.join(base_img_dir, img), os.path.join(base_split_dir, 'test', str(identity), img))

    # Convert labels to one-hot encoding
    labels = {k: tf.one_hot(v, num_ids) for k, v in labels.items()}

    def load_and_preprocess_image(file_path, label):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = tf.keras.applications.vgg16.preprocess_input(image)
        label = tf.cast(label, tf.int32)
        return image, label

    # Instantiate synthetic augmentation layer
    synthetic_augmentation = augmentation_layer.RandomOcclusionLayer(
        augmentation_prob=0.4,  # with 40% chance of synthetic occlusion
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
                                .batch(batch_size),
        'val': tf.data.Dataset.from_tensor_slices((tf.constant(image_paths['val']), labels['val']))
                               .map(load_and_preprocess_image)
                               .batch(batch_size),
        'test_original': tf.data.Dataset.from_tensor_slices((tf.constant(image_paths['test']), labels['test']))
                                      .map(load_and_preprocess_image)
                                      .batch(batch_size),
        'test_augmented': tf.data.Dataset.from_tensor_slices((tf.constant(image_paths['test']), labels['test']))
                                       .map(load_and_preprocess_image)
                                       .map(augment_image)
                                       .batch(batch_size)
    }

    return datasets['train'], datasets['val'], datasets['test_original'], datasets['test_augmented'], num_ids

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

            shutil.move(src_path, dst_path)
            # shutil.copy2(src_path, dst_path)

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

    # # Convert labels to integers
    # labels = {k: tf.cast(tf.constant(v), tf.int32) for k, v in labels.items()}
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
    
# Load CelebA dataset
train_dataset, validation_dataset, test_dataset_original, test_dataset_augmented, num_classes = prepare_celeba_dataset('face_recognition/datasets/celeb_a/img_align_celeba')

# Load VGG16 base model with VGGFace weights (without top - for fine tuning)
base_model = tf.keras.applications.VGG16(weights=None,
                                         include_top=False, input_shape=(224, 224, 3))
weights_path = 'face_recognition/src/global_features/rcmalli_vggface_tf_notop_vgg16.h5'
base_model.load_weights(weights_path)

# Freeze base model layers (optional)
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers with L1 and L2 regularization
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation='relu', 
                          kernel_regularizer=tf.keras.regularizers.l2(0.00001),
                          bias_regularizer=tf.keras.regularizers.l1(0.00001))(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.00001),
                                    bias_regularizer=tf.keras.regularizers.l1(0.00001))(x)

# Create the final model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'precision', 'recall'])

# Performance optimization: prefetch and cache data
train_dataset = train_dataset.take(5000).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.take(5000).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset_original = test_dataset_original.take(5000).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset_augmented = test_dataset_augmented.take(5000).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Fine-tune the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7
)

model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate on the original test set
original_test_loss, original_test_accuracy = model.evaluate(test_dataset_original)

# Evaluate on the augmented test set
augmented_test_loss, augmented_test_accuracy = model.evaluate(test_dataset_augmented)

print("Original Test Accuracy:", original_test_accuracy)
print("Augmented Test Accuracy:", augmented_test_accuracy)

# Paths to save model and weights
model_dir = 'face_recognition/src/global_features/model.keras'
weights_dir = 'face_recognition/src/global_features/weights'
weights_path = os.path.join(weights_dir, 'fine_tuned_weights.h5')

# Ensure directories exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

try:
    # Save the entire model
    model.save(model_dir)  # Creates a directory with model structure and weights
    print(f"Model saved successfully at {model_dir}")

    # Save just the weights (smaller file size)
    model.save_weights(weights_path)  # Only saves the weights
    print(f"Weights saved successfully at {weights_path}")
except Exception as e:
    print(f"An error occurred while saving the model or weights: {e}")

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