import tensorflow as tf
from keras import layers
import tensorflow_datasets as tfds
import numpy as np
import cv2
import augmentation

# Preprocess for ResNet50 fine tuning, make (image, label) pairs
def preprocess_image_resnet50(label, image):
    # Resize and normalise
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.resnet50.preprocess_input(image)

    # Switch label, image
    return image, label

def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_labels(ds):
    # Collect all string labels from dataset into an array
    return np.array([label.numpy() for _, label in ds])

def get_num_classes(ds):
    # Extract all identity labels
    classes = ds.map(lambda features: features['identity'])

    # Use a set to find all unique identities
    unique_classes = set()
    for identity in tfds.as_numpy(classes):
        unique_classes.update(identity)
    
    return len(unique_classes)

def load_identities(identity_file):
    identity_dict = {}
    with open(identity_file, 'r') as file:
        for line in file:
            filename, identity = line.strip().split()
            identity_dict[filename] = int(identity)
    return identity_dict

def load_partitions(partition_file):
    partition_dict = {}
    with open(partition_file, 'r') as file:
        for line in file:
            filename, partition = line.strip().split()
            partition_dict[filename] = int(partition)
    return partition_dict

def load_dataset(dataset_type):

    if dataset_type == 'lfw':
        # (ds, train_ds, val_ds, test_ds), metadata
        (ds, train_ds, val_ds, test_ds), metadata = tfds.load(
            'lfw',
            data_dir='face_recognition/datasets/lfw',
            split=['train', 'train[:80%]', 'train[80%:90%]', 'train[90%:]'],
            as_supervised=True,
            with_info=True,
            batch_size=32
        )

        num_classes = 0
        
        train_ds = train_ds.map(
            augmentation.add_occlusion, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(
            augmentation.add_occlusion, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(
            augmentation.add_occlusion, num_parallel_calls=tf.data.AUTOTUNE)

# Load Pre-trained ResNet50
def load_resnet50_model(num_classes, input_shape=(224,224,3)):
    """ Loads a ResNet50 model and adds fine-tuning layers.

    Args:
        input_shape:  Input shape for the model.
        num_classes: Number of classes in dataset used for fine-tuning.

    Returns: 
        keras.Model: The compiled ResNet50 model ready for fine-tuning.
    """
    # Load model
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
        layers.Dense(num_classes, activation='softmax')  # Output for classification
    ])

    return model
