import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers
import finetune_utils
import augmentation
import cv2

# Feature Fusion Model for Landmark-Based SURF + HOG
def get_fusion_model(input_shape, num_classes):
    # Input for Global Features
    input_global = layers.Input(shape=input_shape) 
    x = finetune_utils.load_resnet50_model(input_shape)(input_global)  # Fine-tuned ResNet50 Model

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

### MAIN LOGIC ###
## 1. Load the LFW dataset
(dataset_train, dataset_test), dataset_info = tfds.load(
    'lfw', # Labelled Faces in the Wild: For Face Recognition in Unconstrained Environments
    split=['train[:80%]', 'train[80%:]'],  # Load both training and test splits
    data_dir='face_recognition/lfw',
    shuffle_files=True,  # Shuffle for training randomness
    as_supervised=True,  # Load as (image, label) pairs for supervised learning
    with_info=True  # Access dataset metadata
)

for image, label in dataset_train.take(1): 
    print(image.shape)  # Output the image shape

# ## 2. Preprocessing
# dataset_train = dataset_train.map(
#     augmentation.augment_with_occlusion, num_parallel_calls=tf.data.AUTOTUNE
# ).map(
#     augmentation.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
# )

# # Split test set (occluded and unoccluded)
# dataset_test_occluded = dataset_test.filter(finetune_utils.ds_split_with_occlusion)
# dataset_test_unoccluded = dataset_test.filter(finetune_utils.ds_split_no_occlusion)

# # Preprocess both test sets
# dataset_test_unoccluded = dataset_test_unoccluded.map(
#     finetune_utils.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
# )
# dataset_test_occluded = dataset_test_occluded.map(
#     finetune_utils.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
# )

# ## 3. Batching (Adjust as needed)
# dataset_train = dataset_train.batch(32)
# dataset_test_occluded = dataset_test_occluded.batch(32)
# dataset_test_unoccluded = dataset_test_unoccluded.batch(32)

# ## 4. Fine-tuning
# model = finetune_utils.load_resnet50_model(input_shape=(224, 224, 3)) 

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Adjust as needed
#     loss='sparse_categorical_crossentropy', 
#     metrics=['accuracy']
# )

# model.fit(
#     dataset_train,
#     epochs=10, 
#     validation_data=dataset_test_unoccluded 
# )

# ## 5. Compile and Train (Outline)
# model.compile(optimizer='adam', 
#               loss=tf.losses.SparseCategoricalCrossentropy(), 
#               metrics=['accuracy'])
# model.fit(dataset_train, 
#           epochs=10, 
#           validation_data=dataset_test)  
