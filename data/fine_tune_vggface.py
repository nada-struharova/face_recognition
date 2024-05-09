import tensorflow as tf
import augmentation
import data_utils

def preprocess_image_vggface(image, label):
  image = tf.image.resize(image, (224, 224))  # VGGFace input shape
  image = augmentation.add_occlusion(label, image) # Synthetically occlude images
  return image, label

# 2. Load VGGFace Model (choose variant "VGG16" or "VGG19")
# TODO: get number of classes (num_classes)
model = data_utils.load_vggface_model()

(ds, train_ds, val_ds, test_ds), metadata = data_utils.load_lfw_dataset

train_ds = train_ds.map(preprocess_image_vggface, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess_image_vggface, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_image_vggface, num_parallel_calls=tf.data.AUTOTUNE)

# 4. Compile and Train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              lr=0.001)
model.fit(train_ds,
          validation_data=val_ds,
          epochs=15)  # Adjust epochs

# Stage 2: Unfreezing and further fine-tuning
for layer in model.layers[:-10]:  # Unfreeze all but the last 10
    layer.trainable = True

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])  # Recompile with a possibly lower learning rate

model.fit(train_ds,
          validation_data=val_ds,
          metrics=['accuracy'],
          lr=0.0001) 