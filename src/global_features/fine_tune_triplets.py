import tensorflow as tf
import os
import data_utils
import model_utils
import confusion_matrix as cm

# ---------------- Constants ----------------
# Training Details
MODEL_TYPE = 'vgg16'  # Other options: 'facenet', 'resnet50'
LOSS_FUNC = 'triplet_loss' # Other options: 'sparse_categorical_crossentropy"
BATCH_SIZE = 2

# Directories
BASE_IMG_DIR = 'face_recognition/datasets/celeb_a/img_align_celeba'
MODEL_DIR = 'face_recognition/src/global_features'
WEIGHTS_DIR = os.path.join(MODEL_DIR, 'weights')

# ---------------- Load Data ----------------
# For triplet loss, center loss, contrastive loss
train_triplets, val_triplets, test_triplets = data_utils.generate_triplets_from_dataset(image_dir=BASE_IMG_DIR)

# ---------------- Create and Train Model ----------------
# Load specific model
model = model_utils.load_model(MODEL_TYPE, BATCH_SIZE)

# Check model summary to verify output shape
model.summary()

# Compile loaded model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=LOSS_FUNC,
              metrics=['accuracy'])

# Performance optimization: prefetch and cache data
train_ds = data_utils.prepare_triplet_dataset(train_triplets, batch_size=BATCH_SIZE, shuffle=True)
val_ds = data_utils.prepare_triplet_dataset(val_triplets, batch_size=BATCH_SIZE, shuffle=False)
test_ds = data_utils.prepare_triplet_dataset(test_triplets, batch_size=BATCH_SIZE, shuffle=False)

checkpoint_path = os.path.join(
    WEIGHTS_DIR,
    f'{MODEL_TYPE}_{LOSS_FUNC}_batch{BATCH_SIZE}_epoch{{epoch:02d}}_val_acc{{val_accuracy:.2f}}.weights.h5'
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='max'
)