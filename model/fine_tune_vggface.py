import tensorflow as tf
from keras import Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Custom parameters
nb_class = 2
hidden_dim = 512
batch_size = 32
epochs = 10

# Model architecture
vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
custom_vgg_model = Model(inputs=vgg_model.input, outputs=out)

# Freeze the layers except the last 3 layers
for layer in custom_vgg_model.layers[:-3]:
    layer.trainable = False

# Compile the model
custom_vgg_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare data augmentation configuration
train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Fit the model
checkpointer = ModelCheckpoint(filepath
