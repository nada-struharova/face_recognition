import tensorflow as tf
import os
import data_utils
import model_utils

# Constants
BASE_IMG_DIR = 'face_recognition/datasets/celeb_a/img_align_celeba'
VGG16_WEIGHTS_PATH = 'face_recognition/src/global_features/weights/rcmalli_vggface_tf_notop_vgg16.h5'
MODEL_TYPE = 'vgg16'  # Other options: 'facenet', 'arcface'
MODEL_DIR = 'face_recognition/src/global_features'
WEIGHTS_DIR = os.path.join(MODEL_DIR, 'weights')
BATCH_SIZE = 32

# Load CelebA dataset
train_ds, val_ds, test_ds_og, test_ds_aug, num_classes = data_utils.prepare_celeba_fr(BASE_IMG_DIR)

# Load the selected model
if MODEL_TYPE == 'vgg16':
    model = model_utils.load_vgg16_model(num_classes, VGG16_WEIGHTS_PATH)
    model_name = 'fr_vgg16_model.keras'
    weights_name = 'fr_vgg16_weights.h5'
elif MODEL_TYPE == 'facenet':
    model = model_utils.load_facenet_model(num_classes)
    model_name = 'fr_facenet_model.keras'
    weights_name = 'fr_facenet_weights.h5'
elif MODEL_TYPE == 'arcface':
    model = model_utils.load_arcface_model(num_classes)
    model_name = 'fr_arcface_model.keras'
    weights_name = 'fr_arcface_weights.h5'
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'precision', 'recall'])

# Performance optimization: prefetch and cache data
train_ds = train_ds.take(5000).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.take(5000).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_og = test_ds_og.take(5000).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_aug = test_ds_aug.take(5000).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Fine-tune the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7
)

model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate on the original test set
original_test_loss, original_test_accuracy = model.evaluate(test_ds_og)

# Evaluate on the augmented test set
augmented_test_loss, augmented_test_accuracy = model.evaluate(test_ds_aug)

print("Original Test Accuracy:", original_test_accuracy)
print("Augmented Test Accuracy:", augmented_test_accuracy)

# Paths to save model and weights
model_dir = 'face_recognition/src/global_features/model.keras'
weights_dir = 'face_recognition/src/global_features/weights'
weights_path = os.path.join(weights_dir, 'fine_tuned_weights.h5')

# Ensure directories exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

# Try to save the entire model
try:
    model.save(model_dir)
    print(f"Model saved successfully at {model_dir}")
except Exception as e:
    print(f"An error occurred while saving the model: {e}")

# Try to save the weights separately
try:
    model.save_weights(weights_path)
    print(f"Weights saved successfully at {weights_path}")
except Exception as e:
    print(f"An error occurred while saving the weights: {e}")