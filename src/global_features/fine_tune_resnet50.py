import tensorflow as tf
from keras import layers
import src.dataset_utils as dataset_utils

def lookup_labels(label, image):
    return lookup(label), image

# 2. Load Dataset
(ds, train_ds, val_ds, test_ds) = dataset_utils.load_lfw_dataset()

# Create the StringLookup layer (convert string labels to int for loss function)
lookup = layers.StringLookup(output_mode='int')
# Adapt the vocabulary to the complete training dataset 
lookup.adapt(ds.map(lambda label, _ : label))
num_classes = lookup.vocabulary_size()

# Convert string labels to int labels to fine tune model
train_ds = train_ds.map(lookup_labels, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(lookup_labels, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(lookup_labels, num_parallel_calls=tf.data.AUTOTUNE)

# 4. Load Model
model = dataset_utils.load_resnet50_model(num_classes=num_classes)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  patience=5,
                                                  restore_best_weights=True)

# 5. Compile and Fine-Tune
model.compile(optimizer='adam', 
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.fit(train_ds, 
          epochs=20,
          validation_data=val_ds,
          callbacks=[early_stopping])
