import tensorflow as tf
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# Load a pre-trained ResNet-50 model (trained on ImageNet)
base_model = ResNet50(weights='imagenet', include_top=False)

# Freeze most layers for fine-tuning
for layer in base_model.layers[:-4]:  # Fine-tune last few layers
    layer.trainable = False

# Add custom layers on top of pre-trained model
""" 1st approach"""
x = base_model.output
x = tf.keras.layers.Flatten()(x)  # Flatten for dense layers 
predictions = Dense(10, activation='softmax')(x)  # Adjust '10' to number of classes in dataset

""" 2nd approach """
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x) 
# predictions = Dense(10, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data generators for training and validation
train_data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Random rotations up to 20 degrees
    width_shift_range=0.2,  # Random horizontal shifts up to 20%
    height_shift_range=0.2,  # Random vertical shifts up to 20%
    shear_range=0.2,# Random shear transformations
    zoom_range=0.2, # Random zooms
    horizontal_flip=True)

test_data_generator = ImageDataGenerator(rescale=1./255)

# Update paths for augmented dataset
train_dataset_path = '/path/to/train/dataset'
validation_dataset_path = '/path/to/validation/dataset'

train_generator = train_data_generator.flow_from_directory(
    train_dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_data_generator.flow_from_directory(
    validation_dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Train the model on the new data
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

# Save the trained model
model.save('finetuned_resnet50.h5')
