import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers
import augmentation
import data_utils

# Preprocess for ResNet50 fine tuning
def preprocess_image_resnet50(label, image):
    # Resize the image to match the input size of ResNet50
    image = tf.image.resize(image, (224, 224))
    # Normalize pixel values to [-1, 1]
    image = tf.keras.applications.resnet50.preprocess_input(image)
    # Encode string label
    label = lookup(label)

    # Switch label, image -> ResNet50 generally expects (inputs, targets) structure
    return image, label

# Create the StringLookup layer (convert string labels to int for loss function)
lookup = tf.keras.layers.StringLookup(output_mode='int')

(ds, train_ds, val_ds, test_ds), metadata = data_utils.load_lfw_dataset()

# Adapt the vocabulary to the complete training dataset 
lookup.adapt(ds.map(lambda label, _ : label))
num_classes = lookup.vocabulary_size()

# 2. Preprocess datasets
train_ds = train_ds.map(
    augmentation.add_occlusion, num_parallel_calls=tf.data.AUTOTUNE).map(
    preprocess_image_resnet50, num_parallel_calls=tf.data.AUTOTUNE)

test_ds = test_ds.map(
    augmentation.add_occlusion, num_parallel_calls=tf.data.AUTOTUNE).map(
    preprocess_image_resnet50, num_parallel_calls=tf.data.AUTOTUNE)

val_ds = val_ds.map(
    augmentation.add_occlusion, num_parallel_calls=tf.data.AUTOTUNE).map(
    preprocess_image_resnet50, num_parallel_calls=tf.data.AUTOTUNE)

## 3. Fine-tuning
model = data_utils.load_resnet50_model(num_classes=num_classes, input_shape=(224, 224, 3))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

## 4. Compile and Fine-Tune
model.compile(optimizer='adam', 
              loss=tf.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])
model.fit(train_ds, 
          epochs=15,
          validation_data=val_ds,
          callbacks=[early_stopping])
