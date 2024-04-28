import tensorflow as tf
from keras import layers
import random
import augmentation
import numpy as np

# Preprocess (General)
def preprocess_image(image, label):
    print(image.shape)
    image = tf.image.resize(image, [224, 224])  # Adjust size as needed
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    return image, label

# Preprocess with Keras Layers
# TODO: can add these layers to our model for less functions and imports
def preprocess_image_krs_layers(image, label):
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(224, 224),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal") 
    ])

    image = resize_and_rescale(image)
    return image, label

def get_labels(ds):
    # Collect all string labels from dataset into an array
    return np.array([label.numpy() for _, label in ds])

# Augmentation with Synthetic Occlusion (using augmentation.py)
def augment_with_occlusion(label, image):
    image = augmentation.add_occlusion(image)  # Augment the image
    return label, image
