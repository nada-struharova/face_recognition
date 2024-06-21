import tensorflow as tf
from keras_facenet import FaceNet
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import dataset_utils

def load_model(model_type, num_classes):
    # Load the selected model
    if model_type == 'vgg16':
        model = load_vgg16_model_categorical(num_classes)
    elif model_type == 'facenet':
        model = load_facenet_model(num_classes)
    elif model_type == 'resnet50':
        model = load_resnet50_model_categorical(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
    
# ----------------- Loading Model -----------------
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
    if len(x.shape) == 2:
        x = tf.keras.layers.Reshape((1, 1, x.shape[1]))(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu', 
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                              bias_regularizer=tf.keras.regularizers.l1(0.0001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                        bias_regularizer=tf.keras.regularizers.l1(0.0001))(x)

    # Create the final model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    
    return model

# Load Pre-Trained VGG16
def load_vgg16_model_categorical(num_ids,
                     include_top=False,
                     weights_path='face_recognition/src/global_features/weights/rcmalli_vggface_tf_notop_vgg16.h5'):
    # Load VGG16 base model with VGGFace weights (without top - for fine tuning)
    base_model = tf.keras.applications.VGG16(weights=None,
                                             include_top=include_top,
                                             input_shape=(224, 224, 3))
    base_model.load_weights(weights_path)

    # Freeze base model layers (optional)
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom top layers with L1 and L2 regularization
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu', 
                              kernel_regularizer=tf.keras.regularizers.l2(0.001),
                              bias_regularizer=tf.keras.regularizers.l1(0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(num_ids, activation='softmax',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                        bias_regularizer=tf.keras.regularizers.l1(0.001))(x)

    # Create the final model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    return model

# Load Pre-Trained VGG16 for extracting global features
def load_vgg16_model_extract(input_shape=(224, 224, 3),
                             include_top=False,
                             weights_path='face_recognition/src/global_features/weights/rcmalli_vggface_tf_notop_vgg16.h5'):
    base_model = tf.keras.applications.VGG16(weights=None,
                                             include_top=include_top,
                                             input_shape=input_shape)
    base_model.load_weights(weights_path)

    for layer in base_model.layers:
        layer.trainable = False

    return base_model

# Load the model with triplet loss and local features
def load_combined_model(base_model, embedding_size=128,
                        local_feature_dim=256):
    x = base_model.output
    
    local_features_input = tf.keras.layers.Input(shape=(local_feature_dim,), name='local_features_input')
    combined_features = tf.keras.layers.Concatenate()([x, local_features_input])
    combined_features = tf.keras.layers.Dense(512, activation='relu',
                                              kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                              bias_regularizer=tf.keras.regularizers.l1(0.001))(combined_features)
    combined_features = tf.keras.layers.Dropout(0.5)(combined_features)
    embeddings = tf.keras.layers.Dense(embedding_size, activation=None, name='embeddings')(combined_features)

    model = tf.keras.models.Model(inputs=[base_model.input, local_features_input], outputs=embeddings)
    return model

# Load ResNet50 Model
def load_resnet50_model_categorical(num_classes,
                        include_top=False,
                        weights_path='face_recognition/src/global_features/weights/rcmalli_vggface_tf_notop_resnet50.h5'):
    # Load ResNet50 base model with pre-trained weights
    base_model = tf.keras.applications.ResNet50(weights=None, 
                                                include_top=include_top, 
                                                input_shape=(224, 224, 3))
    
    # If weights_path is provided, load custom weights
    if include_top==True:
        weights_path = 'face_recognition/src/global_features/weights/rcmalli_vggface_tf_resnet50.h5'
    
    base_model.load_weights(weights_path)

    # Freeze base model layers (optional)
    for layer in base_model.layers:
        layer.trainable = False

    # Get last layer
    # x = base_model.output
    x = base_model.get_layer('avg_pool').output

    # Add custom layers with pooling, dropout regularisation
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu', 
                              kernel_regularizer=tf.keras.regularizers.l2(0.001),
                              bias_regularizer=tf.keras.regularizers.l1(0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                        bias_regularizer=tf.keras.regularizers.l1(0.001))(x)

    # Create the final model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    
    return model

# ----------------- Evaluate -----------------
# Thresholding model predictions
def evaluate_model_with_threshold(model, dataset, threshold=0.8):
    correct_predictions = 0
    total_predictions = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    unknown_predictions = 0
    
    for images, labels in dataset:
        predictions = model.predict(images)
        for i in range(len(predictions)):
            prediction = predictions[i]
            label = labels[i]
            max_prob = np.max(prediction)
            
            if max_prob < threshold:
                unknown_predictions += 1
            else:
                predicted_label = np.argmax(prediction)
                true_label = np.argmax(label)
                
                if predicted_label == true_label:
                    correct_predictions += 1
                    true_positives += 1
                else:
                    if predicted_label != 0:  # Predicted label is not unknown
                        false_positives += 1
                    if true_label != 0:  # True label is not unknown
                        false_negatives += 1
                        
                total_predictions += 1
    
    accuracy = (correct_predictions + unknown_predictions) / (total_predictions + unknown_predictions) if (total_predictions + unknown_predictions) > 0 else 0.0
    accuracy_excluding_unknown = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy_excluding_unknown, accuracy, precision, recall, f1
