import tensorflow as tf
import os
import dataset_utils as dataset_utils
import model_utils as model_utils
import confusion_matrix as cm

# ---------------- Constants ----------------
# Training Details
MODEL_TYPE = 'vgg16'  # Other options: 'facenet', 'resnet50'
LOSS_FUNC = 'categorical_crossentropy' # Other options: 'sparse_categorical_crossentropy'
BATCH_SIZE = 32

# Directories
BASE_IMG_DIR = 'face_recognition/datasets/celeb_a/img_align_celeba_cropped'
MODEL_DIR = 'face_recognition/src/global_features'
WEIGHTS_DIR = os.path.join(MODEL_DIR, 'weights')

model_name = f'{MODEL_TYPE}_{LOSS_FUNC}_{BATCH_SIZE}.keras'
weights_name = f'{MODEL_TYPE}_{LOSS_FUNC}_{BATCH_SIZE}_weights.h5'

# ---------------- Model Training ----------------
# Load CelebA dataset
train_ds, val_ds, test_ds_og, test_ds_aug, num_classes = dataset_utils.prepare_celeba_fr(BASE_IMG_DIR,
                                                                                      loss_func=LOSS_FUNC,
                                                                                      batch_size=BATCH_SIZE)

# Performance optimization: prefetch and cache data
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_og = test_ds_og.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_aug = test_ds_aug.prefetch(buffer_size=tf.data.AUTOTUNE)

# Load specific model
model = model_utils.load_model(MODEL_TYPE, num_classes)

# Check model summary to verify output shape
model.summary()

# Compile loaded model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=LOSS_FUNC,
              metrics=['accuracy', 'precision', 'recall'])

# Early Stopping Callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=10,
                                                  restore_best_weights=True)

# Dynamic Reduction of Learning Rate
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-7)

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

# Display calculated metrics after each training epoch
metrics_callback = cm.MetricsCallback(validation_data=val_ds)

# history = model.fit(
#     train_ds,
#     epochs=10,
#     validation_data=val_ds,
#     callbacks=[metrics_callback, early_stopping, reduce_lr, checkpoint]
# )

# ---------------- Model Evaluation ----------------
# Load best weights after training
# best_weights = max(history.history['val_accuracy'])
# best_epoch = history.history['val_accuracy'].index(best_weights) + 1
# best_weights_path = os.path.join(WEIGHTS_DIR, f'vgg16_epoch:{best_epoch:02d}_val_acc:{best_weights:.2f}.weights.h5')

best_weights_path = 'face_recognition/src/global_features/weights/vgg16_categorical_crossentropy_bs32_epoch:51_val_acc:0.74.weights.h5'
if os.path.exists(best_weights_path):
    model.load_weights(best_weights_path)
    print(f"Weights successfully loaded from: {best_weights_path}")
else:
    print(f"Weights file not found: {best_weights_path}")

# ---------------- Evaluate on Test set (No Augmentation) ----------------
og_results = model.evaluate(test_ds_og)

# Unpack the results correctly
original_test_loss = og_results[0]  # Loss value
original_test_accuracy = og_results[1]  # Accuracy value
original_test_precision = og_results[2]  # Precision value
original_test_recall = og_results[3]  # Recall value

# Print the results
print(f"Original Test Loss: {original_test_loss}")
print(f"Original Test Accuracy: {original_test_accuracy}")
print(f"Original Test Precision: {original_test_precision}")
print(f"Original Test Recall: {original_test_recall}")

# ---------------- Evaluate on Augmented Test Set ----------------
aug_results = model.evaluate(test_ds_aug)

aug_test_loss = aug_results[0]
aug_test_accuracy = aug_results[1]
aug_test_precision = aug_results[2]
aug_test_recall = aug_results[3]

print(f"Augmented Test Loss: {aug_test_loss}")
print(f"Augmented Test Accuracy: {aug_test_accuracy}")
print(f"Augmented Test Precision: {aug_test_precision}")
print(f"Augmented Test Recall: {aug_test_recall}")

# ---------------- Evaluate with Threshold on the Predictions ----------------
# Evaluate on the original test set with threshold
test_acc_no_unknown, test_acc, test_precision, test_recall, test_f1 = model_utils.evaluate_model_with_threshold(model, test_ds_og)

print("Original Test Accuracy with Threshold, excluding unknowns: ", test_acc_no_unknown)
print("Original Test Accuracy with Threshold:", test_acc)
print("Original Test Precision with Threshold:", test_precision)
print("Original Test Recall with Threshold:", test_recall)
print("Original Test F1 Score with Threshold:", test_f1)

# Evaluate on the augmented test set with threshold
aug_test_acc_no_unknown, aug_test_acc, aug_test_precision, aug_test_recall, aug_test_f1 = model_utils.evaluate_with_threshold(model, test_ds_aug)
print("Augmented Test Accuracy with Threshold, excluding unknowns: ", test_acc_no_unknown)
print("Augmented Test Accuracy with Threshold:", aug_test_acc)
print("Augmented Test Precision with Threshold:", aug_test_precision)
print("Augmented Test Recall with Threshold:", aug_test_recall)
print("Augmented Test F1 Score with Threshold:", aug_test_f1)

# ---------------- Save Model ----------------
# Paths to save model and weights
final_model_path = os.path.join(MODEL_DIR, model_name)
final_weights_path = os.path.join(WEIGHTS_DIR, weights_name)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Save Model
try:
    model.save(final_model_path)
    print(f"Model saved successfully at {final_model_path}")
except Exception as e:
    print(f"An error occurred while saving the model: {e}")

# Save Weights
try:
    model.save_weights(final_weights_path)
    print(f"Weights saved successfully at {final_weights_path}")
except Exception as e:
    print(f"An error occurred while saving the weights: {e}")