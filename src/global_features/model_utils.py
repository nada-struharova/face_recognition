import tensorflow as tf
from keras import layers
from keras_facenet import FaceNet
from insightface.model_zoo import get_model

# Load Pre-Trained ResNet50
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

# Load Pre-Trained FaceNet
def load_facenet_model(num_classes):
    # Load FaceNet model
    facenet_model = FaceNet()
    
    # Get the base model (Inception ResNet V1) and remove the top layers
    base_model = facenet_model.model
    
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
    
    return model

# Load Pre-Trained VGG16
def load_vgg16_model(num_classes, weights_path):
    # Load VGG16 base model with VGGFace weights (without top - for fine tuning)
    base_model = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
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
    
    return model

# Load ArcFace
def load_arcface_model(num_classes):
    # Load ArcFace model
    arcface_model = get_model('arcface_r100_v1')
    arcface_model.prepare(ctx_id=-1)  # Use -1 for CPU, set to GPU ID for GPU

    # Get the base model and remove the top layers
    base_model = arcface_model.model
    
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
    
    return model


# ------ OLD ------
# # Load VGG16 base model with VGGFace weights (without top - for fine tuning)
# base_model = tf.keras.applications.VGG16(weights=None,
#                                          include_top=False, input_shape=(224, 224, 3))
# weights_path = 'face_recognition/src/global_features/weights/rcmalli_vggface_tf_notop_vgg16.h5'
# base_model.load_weights(weights_path)

# # Freeze base model layers (optional)
# for layer in base_model.layers:
#     layer.trainable = False

# # Add custom top layers with L1 and L2 regularization
# x = base_model.output
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(512, activation='relu', 
#                           kernel_regularizer=tf.keras.regularizers.l2(0.00001),
#                           bias_regularizer=tf.keras.regularizers.l1(0.00001))(x)
# x = tf.keras.layers.Dropout(0.5)(x)
# predictions = tf.keras.layers.Dense(num_classes, activation='softmax',
#                                     kernel_regularizer=tf.keras.regularizers.l2(0.00001),
#                                     bias_regularizer=tf.keras.regularizers.l1(0.00001))(x)

# # Create the final model
# model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)