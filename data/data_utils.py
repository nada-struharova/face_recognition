import tensorflow as tf
from keras import layers
import tensorflow_datasets as tfds
import numpy as np
import cv2
import augmentation
import os
import pandas as pd
from PIL import Image
from pathlib import Path

# Preprocess (General)
def preprocess_image(image, label):
    image = tf.image.resize(image, [224, 224])  # Adjust size as needed
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    return image, label

# Preprocess with Keras Layers
def preprocess_image_krs_layers(image, label):
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(224, 224),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal") 
    ])

    image = resize_and_rescale(image)
    return image, label

# Preprocess for ResNet50 fine tuning, make (image, label) pairs
def preprocess_image_resnet50(label, image):
    # Resize and normalise
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.resnet50.preprocess_input(image)

    # Switch label, image
    return image, label

def preprocess_image_vgg16(label, image):
    # Resize and normalise
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.vgg16.preprocess_input(image)

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

def load_and_process_image(image_file, identity):
    image = tf.io.read_file('face_recognition/datasets/celeb_a/img_align_celeba' + "/" + image_file)
    image = tf.image.decode_jpeg(image, channels=3)

    # *** Apply augmentation ***
    image = augmentation.add_occlusion(identity, image) 

    # *** Apply preprocessing ***
    image = tf.image.resize(image, [224, 224])  # Example: Resize to 224x224
    image = tf.keras.applications.imagenet_utils.preprocess_input(image)  # Example: ImageNet preprocessing

    return image, identity

def preprocess(data):
    """ Preprocess the data: extract image and identity label. """
    image = data['image']
    identity = data['attributes']['identity']
    return identity, image

def create_dataset(image_files, split, partitions, identities):
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.filter(lambda img_file: partitions[img_file] == split.numpy())
    dataset = dataset.map(
        lambda image_file: load_and_process_image(image_file, identities[image_file]), 
        num_parallel_calls=tf.data.AUTOTUNE 
     )

    # *** Configure dataset properties ***
    dataset = dataset.shuffle(1024)  # Adjust shuffle buffer size 
    dataset = dataset.batch(32)  # Adjust batch size
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

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

    elif dataset_type == 'celeb_a':

        (ds, train_df, val_df, test_df), num_classes = load_dataset_as_df()

        train_ds = prepare_dataset(train_df)
        val_ds = prepare_dataset(val_df)
        test_ds = prepare_dataset(test_df)

    return (ds, train_ds, val_ds, test_ds), num_classes

def load_dataset_from_dir():
    # Load identity and partition file
    identity_dict = load_identities('face_recognition/datasets/celeb_a/identity_CelebA.txt')
    partition_dict = load_partitions('face_recognition/datasets/celeb_a/list_eval_partition.txt')
    image_dir = 'face_recognition/datasets/celeb_a/img_align_celeba'

    # Get number of classes
    unique_ids = set(identity_dict.values())
    num_classes = len(unique_ids)

    images = []
    identities = []
    partitions = []

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            filepath = os.path.join(image_dir, filename)
            image = tf.io.decode_jpeg(tf.io.read_file(filepath))
            image = tf.image.resize(image, [178, 218])  # Resize if necessary

            identity = identity_dict[filename]
            partition = partition_dict[filename]

            images.append(image)
            identities.append(identity)
            partitions.append(partition)
    
    ds = tf.data.Dataset.from_tensor_slices((identities, images, partitions))
    ds = ds.map(lambda identity, image, partition: (augmentation.add_occlusion(identity, image), partition))

    train_ds = ds.filter(lambda x, partition: partition == 0).map(lambda x, partition: x)
    val_ds = ds.filter(lambda x, partition: partition == 1).map(lambda x, partition: x)
    test_ds = ds.filter(lambda x, partition: partition == 2).map(lambda x, partition: x)

    return (ds, train_ds, val_ds, test_ds), num_classes

def load_dataset_as_df():
    image_dir='face_recognition/datasets/celeb_a/img_align_celeba'
    identity_file = 'face_recognition/datasets/celeb_a/identity_CelebA.txt'
    partition_file = 'face_recognition/datasets/celeb_a/list_eval_partition.txt'

    # Load the identity file into a pandas DataFrame
    df_identity = pd.read_csv(identity_file,
                              sep=" ",
                              header=None,
                              names=['filename', 'identity'])
    df_partition = pd.read_csv(partition_file,
                               sep=" ",
                               header=None,
                               names=['filename', 'partition'])
    # Count unique identities directly with pandas
    num_classes = df_identity['identity'].nunique()

    # Merge identity and partition dataframes on filename
    df = pd.merge(df_identity, df_partition, on='filename')

    def load_and_occlude_image(filename, image_dir):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)  # Load with OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR (cv2) to RGB conversion
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = augmentation.occlude(image)  # Apply your occlusion function
        return image
    
        # image_path = os.path.join(image_dir, filename)
        # image = tf.io.read_file(image_path)
        # image = tf.image.decode_jpeg(image, channels=3)  # Could be PNG, adjust accordingly
        # image = augmentation.occlude(image)  # Apply your occlusion function
        # return image
        
    # Add a column for loading and occluding images
    print("before occlusion")
    df['image'] = df['filename'].apply(lambda filename: load_and_occlude_image(filename, image_dir))
    print("after_occlusion")

    # Split the dataframe based on the partition column
    train_df = df[df['partition'] == 0]
    val_df = df[df['partition'] == 1]
    test_df = df[df['partition'] == 2]

    return (df, train_df, val_df, test_df), num_classes 
    

def prepare_dataset(df, batch_size=32):
    # Convert DataFrame columns to a Dataset of tensors
    dataset = tf.data.Dataset.from_tensor_slices((list(df['image']), list(df['identity'])))
    # Optimisation: Batch and Prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
    
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

# Load Pre-Trained VGG16/VGG19 model
def load_vgg_model(num_classes,
                       variant="VGG16",
                       input_shape=(224, 224, 3),
                       include_top=False, 
                       unfreeze_layers=5):
    """ Loads a VGG model (VGG16 by default) and adds fine-tuning layers.

    Args:
        variant:  "VGG16" or "VGG19"
        num_classes: Number of classes in dataset used for fine-tuning.
        input_shape:  Input shape for the model.
        include_top: Whether to include the original VGG classifier.
        unfreeze_layers:  Number of layers to unfreeze at the end of the model.

    Returns: 
        keras.Model: The compiled VGG model ready for fine-tuning.
    """

    # Select the appropriate base model constructor
    if variant == "VGG16":
        base_model_constructor = tf.keras.applications.vgg16.VGG16 
    elif variant == "VGG19":
        base_model_constructor = tf.keras.applications.vgg19.VGG19 
    else:
        raise ValueError(f"Unsupported VGG variant: {variant}")

    # Load the base model with face-specific weights
    vgg_model = base_model_constructor(
        weights='imagenet', include_top=include_top, input_shape=input_shape
    )

    # Freeze all but the last few layers
    for layer in vgg_model.layers[:-unfreeze_layers]: 
        layer.trainable = False

    # Finetuning Layers
    x = layers.GlobalAveragePooling2D()(vgg_model.output)
    x = layers.Dense(1024, activation='relu')(x) 
    x = layers.Dropout(0.5)(x)  
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    # Fine-Tuning Model 
    return tf.keras.Model(inputs=vgg_model.input, outputs=predictions) 