import tensorflow as tf
from keras import layers
import tensorflow_datasets as tfds
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

def load_lfw_dataset():
    # (ds, train_ds, val_ds, test_ds), metadata
    return tfds.load(
        'lfw',
        data_dir='face_recognition/datasets/lfw',
        split=['train', 'train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
        batch_size=32
    )

def load_celeb_a_dataset():
    # (train_ds, val_ds, test_ds), metadata
    return tfds.load(
        'celeb_a',
        data_dir='face_recognition/datasets/celeb_a',
        split=['train', 'validation', 'test'],
        with_info=True,
        as_supervised=True,
        batch_size=32
    )

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

def load_vggface_model(variant="VGG16",
                       num_classes=10,
                       input_shape=(224, 224, 3),
                       include_top=False, 
                       unfreeze_layers=5):
    """ Loads a VGGFace model (VGG16 by default) and adds fine-tuning layers.

    Args:
        variant:  "VGG16" or "VGG19"
        num_classes: Number of classes in dataset used for fine-tuning.
        input_shape:  Input shape for the model.
        include_top: Whether to include the original VGGFace classifier.
        unfreeze_layers:  Number of layers to unfreeze at the end of the model.

    Returns: 
        keras.Model: The compiled VGGFace model ready for fine-tuning.
    """
  
    # Select the appropriate base model constructor
    if variant == "VGG16":
        base_model_constructor = tf.keras.applications.vgg16.VGG16 
    elif variant == "VGG19":
        base_model_constructor = tf.keras.applications.vgg19.VGG19 
    else:
        raise ValueError(f"Unsupported VGGFace variant: {variant}")

    # Load the base model with face-specific weights
    base_model = base_model_constructor(
        weights='vggface', include_top=include_top, input_shape=input_shape
    )

    # Freeze all but the last few layers
    for layer in base_model.layers[:-unfreeze_layers]: 
        layer.trainable = False

    # Add customization layers using Sequential style
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

