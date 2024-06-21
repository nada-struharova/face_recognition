import tensorflow as tf
import model_utils
import dataset_utils
import confusion_matrix as cm
import custom_layers as cl
import os
import numpy as np

# ------------------ Define Constants ------------------
MODEL_TYPE = 'vgg16'  # Options: 'vgg16', 'facenet', 'resnet50'
LOSS_FUNC = 'categorical_crossentropy'
BATCH_SIZE = 32

# Directories
BASE_IMG_DIR = 'face_recognition/datasets/celeb_a/img_align_celeba_cropped'
MODEL_DIR = 'face_recognition/src/global_features'
IDENTITY_FILE = 'face_recognition/datasets/celeb_a/identity_CelebA.txt'
WEIGHTS_DIR = os.path.join(MODEL_DIR, 'weights')

model_name = f'{MODEL_TYPE}_{LOSS_FUNC}_{BATCH_SIZE}.keras'
weights_name = f'{MODEL_TYPE}_{LOSS_FUNC}_{BATCH_SIZE}_weights.h5'

# ------------------ Define Models to Compare ------------------
# define the global model
def define_global_model(base_model, num_classes):
    # Add Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)  # Pool after base_model

    # Add classification layers on top of the features
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x) 
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  

    # Create the model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

    return model

def define_combined_model(base_model, num_classes, num_sift_features=500):
    # Input for the original image
    input_image = tf.keras.layers.Input(shape=(224, 224, 3), name="input_image")

    # Preprocessing for VGG16
    vgg16_preprocessed = cl.PreprocessVGG16()(input_image)

    # Global Feature Extraction (VGG16)
    vgg16_features = base_model(vgg16_preprocessed)
    vgg16_features = tf.keras.layers.GlobalAveragePooling2D()(vgg16_features)

    # Preprocessing for SIFT
    sift_preprocessed = cl.PreprocessGrayscale()(input_image)
    local_features = cl.LocalFeatureLayer()(sift_preprocessed)
    sift_features = tf.keras.layers.GlobalAveragePooling2D()(local_features)

    # Combine Features
    combined_features = tf.keras.layers.Concatenate()([vgg16_features, sift_features])

    # Classification Layers
    x = tf.keras.layers.Dense(512, activation='relu')(combined_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = tf.keras.models.Model(inputs=input_image, outputs=output)

    return model


# Function to load the VGG16 model and define the global model
def load_global_model(num_classes):
    base_model = model_utils.load_vgg16_model_extract()
    global_model = define_global_model(base_model, num_classes)
    
    # Compile the global model with triplet loss
    global_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                         loss=LOSS_FUNC,
                         metrics = ['accuracy'])
    
    return global_model

def load_combined_model(num_classes, local_feature_dim=640, embedding_size=128):
    # Load pre-trained base model
    base_model = model_utils.load_vgg16_model_extract()

    # Define the combined model
    model = define_combined_model(base_model, num_classes, local_feature_dim)

    # Compile the model
    model.compile(optimizer=tf.keras.oprimizers.Adam(learning_rate=0.00001),
                  loss_func=LOSS_FUNC,
                  metrics=['accuracy'])

    return model

# ---------------- Training ----------------
# Load CelebA dataset
train_ds, val_ds, test_ds_og, test_ds_aug, num_classes = dataset_utils.prepare_celeba_fr(BASE_IMG_DIR,
                                                                                      loss_func=LOSS_FUNC,
                                                                                      batch_size=BATCH_SIZE)

# Performance optimization: prefetch and cache data
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_og = test_ds_og.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_aug = test_ds_aug.prefetch(buffer_size=tf.data.AUTOTUNE)

global_model = load_combined_model(num_classes)

# Model Checkpoint: saves best weights
checkpoint_path = os.path.join(
    WEIGHTS_DIR,
    f'{MODEL_TYPE}_{LOSS_FUNC}_batch{BATCH_SIZE}_epoch{{epoch:02d}}_val_acc{{val_accuracy:.2f}}.weights.h5'
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                monitor='val_accuracy',
                                                verbose=1, 
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode='max')

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc',
                                         patience=10,
                                         restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                factor=0.5,
                                                patience=3,
                                                min_lr=1e-7)

metrics_callback = cm.MetricsCallback(val_ds)

# Train the model
global_model.fit(
    train_ds,
    epochs=10
)

history = global_model.fit(
    train_ds,
    epochs=75, 
    callbacks=[metrics_callback, early_stopping, reduce_lr, checkpoint],
    validation_data=val_ds
)

# ---------------- Model Evaluation ----------------
# Load best weights after training
best_weights = max(history.history['val_accuracy'])
best_epoch = history.history['val_accuracy'].index(best_weights) + 1
best_weights_path = os.path.join(WEIGHTS_DIR, f'vgg16_epoch:{best_epoch:02d}_val_acc:{best_weights:.2f}.weights.h5')

if os.path.exists(best_weights_path):
    global_model.load_weights(best_weights_path)
    print(f"Weights successfully loaded from: {best_weights_path}")
else:
    print(f"Weights file not found: {best_weights_path}")

# Evaluate on the original test set with threshold
test_acc_no_unknown, test_acc, test_precision, test_recall, test_f1 = model_utils.evaluate_model_with_threshold(global_model, test_ds_og)

print("Original Test Accuracy with Threshold, excluding unknowns: ", test_acc_no_unknown)
print("Original Test Accuracy with Threshold:", test_acc)
print("Original Test Precision with Threshold:", test_precision)
print("Original Test Recall with Threshold:", test_recall)
print("Original Test F1 Score with Threshold:", test_f1)

# Evaluate on the augmented test set with threshold
aug_test_acc_no_unknown, aug_test_acc, aug_test_precision, aug_test_recall, aug_test_f1 = model_utils.evaluate_with_threshold(global_model, test_ds_aug)
print("Augmented Test Accuracy with Threshold, excluding unknowns: ", test_acc_no_unknown)
print("Augmented Test Accuracy with Threshold:", aug_test_acc)
print("Augmented Test Precision with Threshold:", aug_test_precision)
print("Augmented Test Recall with Threshold:", aug_test_recall)
print("Augmented Test F1 Score with Threshold:", aug_test_f1)

# Evaluate on the original test set
original_test_loss, original_test_accuracy = global_model.evaluate(test_ds_og)
# Evaluate on the augmented test set
augmented_test_loss, augmented_test_accuracy = global_model.evaluate(test_ds_aug)

print("Original Test Accuracy:", original_test_accuracy)
print("Augmented Test Accuracy:", augmented_test_accuracy)

# ---------------- Save Model ----------------
# Paths to save model and weights
final_model_path = os.path.join(MODEL_DIR, model_name)
final_weights_path = os.path.join(WEIGHTS_DIR, weights_name)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Save Model
try:
    global_model.save(final_model_path)
    print(f"Model saved successfully at {final_model_path}")
except Exception as e:
    print(f"An error occurred while saving the model: {e}")

# Save Weights
try:
    global_model.save_weights(final_weights_path)
    print(f"Weights saved successfully at {final_weights_path}")
except Exception as e:
    print(f"An error occurred while saving the weights: {e}")

# # ------------------- Visualise for report -------------------
# import matplotlib.pyplot as plt
# import region_utils

# # Example landmarks
# grid = False

# image_path = 'face_recognition/datasets/celeb_a/img_align_celeba_cropped/110369.jpg'
# image = tf.io.read_file(image_path)
# image = tf.image.decode_jpeg(image, channels=3)
# image = tf.image.resize(image, (224, 224))
# landmarks = {
#     'left_eye': (59, 80),
#     'right_eye': (164, 80),
#     'nose': (112, 126),
#     'mouth_left': (68, 169),
#     'mouth_right': (152, 171)
# }

# if not grid:
#     regions = region_utils.split_to_regions(image, landmarks)
# if grid:
#     regions = region_utils.split_tensor_to_grid(image)
# region_utils.visualize_regions(image, regions, landmarks, grid)