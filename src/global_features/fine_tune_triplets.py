import tensorflow as tf
import os
import data_utils
import model_utils
import confusion_matrix as cm

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), axis=0)
    return loss

def preprocess_images(image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        save_path = os.path.join(output_dir, img_name)
        data_utils.detect_and_crop_face(img_path, save_path)

# ---------------- Constants ----------------
# Training Details
MODEL_TYPE = 'vgg16'  # Options: 'vgg16', 'facenet', 'resnet50'
LOSS_FUNC = 'triplet_loss'
BATCH_SIZE = 32

# Directories
BASE_IMG_DIR = 'face_recognition/datasets/celeb_a/img_align_celeba'
CROPPED_IMG_DIR = 'face_recognition/datasets/celeb_a/img_align_celeba_cropped'
MODEL_DIR = 'face_recognition/src/global_features'
WEIGHTS_DIR = os.path.join(MODEL_DIR, 'weights')

model_name = f'{MODEL_TYPE}_{LOSS_FUNC}_{BATCH_SIZE}.keras'
weights_name = f'{MODEL_TYPE}_{LOSS_FUNC}_{BATCH_SIZE}_weights.h5'

# Preprocess images
preprocess_images(BASE_IMG_DIR, CROPPED_IMG_DIR)

# ---------------- Load Data ----------------
# Generate Triplets
train_triplets, val_triplets, test_triplets = data_utils.generate_triplets_from_dataset(BASE_IMG_DIR)

# Prepare Datasets
train_ds = data_utils.prepare_triplet_dataset(train_triplets, BATCH_SIZE)
val_ds = data_utils.prepare_triplet_dataset(val_triplets, BATCH_SIZE)
test_ds = data_utils.prepare_triplet_dataset(test_triplets, BATCH_SIZE)

# ---------------- Create and Train Model ----------------
# Load specific model
model = model_utils.load_vgg16_model_triplet()

# Check model summary to verify output shape
model.summary()

# Compile loaded model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=triplet_loss)

checkpoint_path = os.path.join(
    WEIGHTS_DIR,
    f'{MODEL_TYPE}_{LOSS_FUNC}_batch{BATCH_SIZE}_epoch{{epoch:02d}}_val_acc{{val_accuracy:.2f}}.weights.h5'
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='min'
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=10,
                                         restore_best_weights=True)
reduce_lr = tf.kers.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.5,
                                                patience=3,
                                                min_lr=1e-7)

# Training
history = model.fit(train_ds,
                    epochs=75,
                    validation_data=val_ds,
                    callbacks=[early_stopping, reduce_lr, checkpoint])

# Load Best Weights
best_weights = min(history.history['val_loss'])
best_epoch = history.history['val_loss'].index(best_weights) + 1
best_weights_path = os.path.join(WEIGHTS_DIR, f'{MODEL_TYPE}_triplet_bs{BATCH_SIZE}_epoch:{best_epoch:02d}_val_loss:{best_weights:.2f}.weights.h5')

if os.path.exists(best_weights_path):
    model.load_weights(best_weights_path)
    print(f"Weights successfully loaded from: {best_weights_path}")
else:
    print(f"Weights file not found: {best_weights_path}")

# Evaluate on the original test set with threshold
test_acc_no_unknown, test_acc, test_precision, test_recall, test_f1 = model_utils.sevaluate_model_with_threshold(model, test_ds_og)

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

# Evaluate Model
test_loss = model.evaluate(test_ds)
print("Test Loss:", test_loss)