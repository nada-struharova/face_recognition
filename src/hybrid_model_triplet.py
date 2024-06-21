import tensorflow as tf
import model_utils
import dataset_utils
import confusion_matrix as cm
import custom_layers as cl
import os
import numpy as np

# ------------------ Define Constants ------------------
MODEL_TYPE = 'vgg16'  # Options: 'vgg16', 'facenet', 'resnet50'
LOSS_FUNC = 'triplet_loss'
BATCH_SIZE = 32

# Directories
BASE_IMG_DIR = 'face_recognition/datasets/celeb_a/img_align_celeba_cropped'
MODEL_DIR = 'face_recognition/src/global_features'
IDENTITY_FILE = 'face_recognition/datasets/celeb_a/identity_CelebA.txt'
WEIGHTS_DIR = os.path.join(MODEL_DIR, 'weights')

# ------------------ Define Loss ------------------
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

# ------------------ Define Models to Compare ------------------
# Function to define the global model
def define_global_model(base_model, embedding_size=128):
    # Define input shape for a single triplet
    triplet_input_shape = (3, 224, 224, 3)
    
    # Define input layer for the triplet
    triplet_input = tf.keras.layers.Input(shape=triplet_input_shape, name='triplet_input')

    # Unpack the triplet into separate images
    triplet_unpack = cl.TripletUnpackLayer()(triplet_input)
    anchor_input, positive_input, negative_input = triplet_unpack

    # Extract embeddings for each image using the base model
    anchor_embedding = base_model(anchor_input)
    positive_embedding = base_model(positive_input)
    negative_embedding = base_model(negative_input)

    # Combine the embeddings into a single output (triplet embeddings)
    x = tf.keras.layers.Concatenate(axis=1)([anchor_embedding, positive_embedding, negative_embedding])
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(embedding_size, activation=None)(x)

    # Create the model
    model = tf.keras.models.Model(inputs=triplet_input, outputs=x)
    
    return model

# # Define combined (local + global) model
# def define_combined_model(base_model, local_feature_dim=256, embedding_size=128):
#     input_global = tf.keras.layers.Input(shape=(224, 224, 3), name='input_global')
#     input_local = tf.keras.layers.Input(shape=(224, 224, 3), name='input_local')

#     # Extract global features
#     preprocessed_global = tf.keras.layers.Lambda(dataset_utils.preprocess_for_vgg16, name='vgg16_preprocessing')(input_global)
#     global_features = base_model(preprocessed_global)

#     # Extract local features
#     grayscale_local = tf.keras.layers.Lambda(tf.image.rgb_to_grayscale, name='grayscale_preprocessing')(input_local)
#     local_features = cl.LocalFeatureLayer()(grayscale_local)

#     # Combine features
#     combined_features = tf.keras.layers.Concatenate()([global_features, local_features])

#     # Train further classification / similarity layers
#     x = tf.keras.layers.Dense(512, activation='relu',
#                               kernel_regularizer=tf.keras.regularizers.l2(0.001),
#                               bias_regularizer=tf.keras.regularizers.l1(0.001))(combined_features)
#     x = tf.keras.layers.Dropout(0.5)(x)
#     embeddings = tf.keras.layers.Dense(embedding_size,
#                                        activation=None,
#                                        name='embeddings')(x)

#     model = tf.keras.models.Model(inputs=[input_global, input_local], outputs=embeddings)
#     return model

# Function to load the VGG16 model and define the global model
def load_global_model_with_triplet_loss():
    base_model = model_utils.load_vgg16_model_extract()
    global_model = define_global_model(base_model)
    
    # Compile the global model with triplet loss
    global_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss=triplet_loss)
    
    return global_model

# Load Combined Model with Triplet Loss
def load_combined_model_with_triplet_loss():
    base_model = model_utils.load_vgg16_model_extract()
    model = define_combined_model(base_model)

    anchor_input = tf.keras.layers.Input(shape=(224, 224, 3), name='anchor_input')
    positive_input = tf.keras.layers.Input(shape=(224, 224, 3), name='positive_input')
    negative_input = tf.keras.layers.Input(shape=(224, 224, 3), name='negative_input')

    anchor_embedding = model([anchor_input, anchor_input])
    positive_embedding = model([positive_input, positive_input])
    negative_embedding = model([negative_input, negative_input])

    combined_output = tf.stack([anchor_embedding, positive_embedding, negative_embedding], axis=1)

    triplet_model = tf.keras.models.Model(inputs=[anchor_input, positive_input, negative_input], outputs=combined_output)
    triplet_model.compile(optimizer='adam', loss=triplet_loss)
    
    return triplet_model

# ---------------- Training ----------------
# Generate triplets
triplets = dataset_utils.generate_triplets()

if triplets:
    # Split triplets into training, validation, and test sets
    train_triplets, val_triplets, test_triplets = dataset_utils.split_triplets(triplets)
    
    # Prepare datasets
    train_ds = dataset_utils.prepare_triplet_dataset(train_triplets, BATCH_SIZE, shuffle=True)
    val_ds = dataset_utils.prepare_triplet_dataset(val_triplets, BATCH_SIZE)
    test_ds = dataset_utils.prepare_triplet_dataset(test_triplets, BATCH_SIZE)
    
triplet_metrics_callback = cm.TripletMetricsCallback(val_triplets)

global_model = load_global_model_with_triplet_loss()

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
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.5,
                                                patience=3,
                                                min_lr=1e-7)

metrics_callback = cm.TripletMetricsCallback(val_ds)


# Train the model (using the anchor_images, positive_images, negative_images)
global_model.fit(
    train_ds,
    epochs=10
)

global_model.fit(
    train_ds,
    epochs=75, 
    callbacks=[triplet_metrics_callback],
    validation_data=val_ds
)

# # ------------------- Visualise for report -------------------
# import matplotlib.pyplot as plt

# # Example landmarks
# grid = True

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
#     regions = split_tensor_to_regions(image, landmarks)
# if grid:
#     regions = split_tensor_to_grid(image)
# visualize_regions(image, regions, landmarks, grid)