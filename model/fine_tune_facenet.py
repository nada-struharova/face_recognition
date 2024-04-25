import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Load pre-trained FaceNet model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(128, activation='sigmoid')(x)  # Assuming 128 is the embedding size
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy')

# Prepare data using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(160, 160),  # Adjust depending on the input size of the FaceNet model
        batch_size=32,
        class_mode='sparse')

validation_generator = val_datagen.flow_from_directory(
        'dataset/val',
        target_size=(160, 160),
        batch_size=32,
        class_mode='sparse')

# Fine-tune the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)
