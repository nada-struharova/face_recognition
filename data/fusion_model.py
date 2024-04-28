import tensorflow as tf
from keras import layers
import data_utils

# Feature Fusion Model for Landmark-Based SURF + HOG
def get_fusion_model(input_shape, num_classes):
    # Input for Global Features
    input_global = layers.Input(shape=input_shape) 
    x = data_utils.load_resnet50_model(input_shape)(input_global)  # Fine-tuned ResNet50 Model

    # Inputs + Processing for Local Features (5 landmarks)
    inputs_local = []
    features = []
    for i in range(5): 
        local_input = layers.Input(shape=(64 + 36,))  # SURF (64) + HOG (36)
        inputs_local.append(local_input)
        processed = layers.Dense(64, activation='relu')(local_input)  # Example processing
        features.append(processed)

    # Concatenate all local features
    combined_local = layers.Concatenate()(features)

    # Fusion of Global + Combined Local Features
    combined = layers.Concatenate()([x, combined_local]) 
    combined = layers.Dense(512, activation='relu')(combined)  

    # Output Layer (adjust as needed)
    output = layers.Dense(num_classes, activation='softmax')(combined)

    model = tf.keras.Model(inputs=[input_global] + inputs_local, outputs=output) 
    return model