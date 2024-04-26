import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers
import random
import augmentation

# Data Preprocessing 
def preprocess_image(image, label):
    image = tf.image.resize(image, [224, 224])  # Adjust size as needed
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    return image, label

# Augmentation with Synthetic Occlusion (using augmentation.py)
def augment_with_occlusion(image, label):
    image = augmentation.add_occlusion(image)  # Augment the image
    return image, label

# Create separate test splits with and without occlusions
def ds_split_with_occlusion(data):
    image, label = data
    if random.random() < 0.5:  # 50% chance of occlusion
        return image, label
    else:
        return None  # Exclude from occluded set

def ds_split_no_occlusion(data):
    image, label = data
    return image, label

# Load Pre-trained ResNet50
def load_resnet50_model(input_shape):
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=input_shape
    )

    # Freeze all but last few model layers (optional)
    for layer in base_model.layers[:-5]:  # work only on last 5 layers
        layer.trainable = False  

    # Custom layers
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(), # For global feature extraction
        layers.Dense(512, activation='relu'),  # Adjust as needed
        # ... can add more layers here ...
        layers.Dense(10, activation='softmax')  # Output for classification
            # TODO: adjust 10 = number of classes
    ])
    return model