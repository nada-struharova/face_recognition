import tensorflow as tf
import random
import matplotlib.pyplot as plt
from keras import layers

class RandomOcclusionLayer(layers.Layer):
    def __init__(self, augmentation_prob, sunglasses_path, hat_path, mask_path, **kwargs):
        super(RandomOcclusionLayer, self).__init__(**kwargs)
        self.augmentation_prob = augmentation_prob
        self.sunglasses = self.load_image(sunglasses_path)
        self.hat = self.load_image(hat_path)
        self.mask = self.load_image(mask_path)
        self.augmentation_images = {
            'sunglasses': self.sunglasses,
            'hat': self.hat,
            'mask': self.mask
        }
        self.scaling_factors = {
            'sunglasses': 0.5,
            'hat': 0.7,
            'mask': 0.5
        }
        self.avg_landmarks = {
            'lefteye_x': 88.85,
            'lefteye_y': 114.57,
            'righteye_x': 135.03,
            'righteye_y': 113.64,
            'nose_x': 114.77,
            'nose_y': 135.74,
            'leftmouth_x': 90.36,
            'leftmouth_y': 156.18,
            'rightmouth_x': 130.62,
            'rightmouth_y': 157.52
        }

    def load_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=4)
        image = tf.image.resize(image, [224, 224])
        return image
    
    def augment_single_image(self, image):
        # Select a random augmentation method
        aug_img_type = random.choice(list(self.augmentation_images.keys()))
        aug_img = self.augmentation_images[aug_img_type]

        # Get appropriate scaling factor
        scaling_factor = self.scaling_factors[aug_img_type]

        # Resize the overlay image based on the scaling factor
        overlay_h = int(aug_img.shape[0] * scaling_factor)
        overlay_w = int(aug_img.shape[1] * scaling_factor)
        aug_img_resized = tf.image.resize(aug_img, [overlay_h, overlay_w])

        if aug_img_type == 'sunglasses':
            overlay_position = self.get_sunglasses_position(self.avg_landmarks, overlay_w, overlay_h)
        elif aug_img_type == 'hat':
            overlay_position = self.get_hat_position(self.avg_landmarks, overlay_w, overlay_h)
        elif aug_img_type == 'mask':
            overlay_position = self.get_mask_position(self.avg_landmarks, overlay_w, overlay_h)
        
        image = self.apply_overlay(image, aug_img_resized, overlay_position)

        # Geometric Transformations
        if random.random() < 0.5:
            image = tf.image.random_crop(image, size=[200, 200, 3])  # Random cropping
            image = tf.image.resize(image, (224, 224))            # Resize back to original size
        if random.random() < 0.5:
            image = tf.image.random_flip_left_right(image)  # Random horizontal flip
        if random.random() < 0.5:
            image = tf.image.rot90(image, k=random.randint(1, 3))  # Random 90-degree rotation

        # Color Transformations
        if random.random() < 0.5:
            image = tf.image.random_brightness(image, max_delta=0.2)  # Random brightness
        if random.random() < 0.5:
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Random contrast
        if random.random() < 0.5:
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  # Random saturation

        return image
    
    def get_sunglasses_position(self, avg_landmarks, overlay_w, overlay_h):
        # Center the sunglasses around the eyes
        center_x = (avg_landmarks['lefteye_x'] + avg_landmarks['righteye_x']) // 2
        center_y = (avg_landmarks['lefteye_y'] + avg_landmarks['righteye_y']) // 2
        position_x = center_x - overlay_w // 2
        position_y = center_y - overlay_h // 2
        return (position_x, position_y)

    def get_hat_position(self, avg_landmarks, overlay_w, overlay_h):
        # Place the hat above the eyes, centered horizontally
        center_x = (avg_landmarks['lefteye_x'] + avg_landmarks['righteye_x']) // 2
        position_x = center_x - overlay_w // 2 + 20
        position_y = avg_landmarks['lefteye_y'] - overlay_h - 100
        return (position_x, position_y)
    
    def get_mask_position(self, avg_landmarks, overlay_w, overlay_h):
        # Center the mask around the nose and mouth, adjusted lower
        center_x = avg_landmarks['nose_x']
        center_y = (avg_landmarks['nose_y'] + avg_landmarks['leftmouth_y']) // 2
        position_x = center_x - overlay_w // 2
        position_y = center_y - overlay_h // 3 + 20  # Adjust the vertical position to be lower
        return (position_x, position_y)

    def apply_overlay(self, background, overlay_image, position):
        bg_h, bg_w, _ = background.shape
        overlay_h, overlay_w, _ = overlay_image.shape
        
        # Adjust position based on resized overlay image
        overlay_x = int(position[0])
        overlay_y = int(position[1])

        # Ensure overlay position is within bounds
        overlay_x = max(0, min(bg_w - overlay_w, overlay_x))
        overlay_y = max(0, min(bg_h - overlay_h, overlay_y))

        # Adjust overlay width if necessary
        overlay_w = min(overlay_w, bg_w - overlay_x)

        # Extract the alpha channel
        alpha_mask = overlay_image[:, :, 3] / 255.0
        alpha_mask = tf.stack([alpha_mask] * 3, axis=-1)  # Create 3-channel alpha mask

        # Extract the RGB channels
        overlay_rgb = overlay_image[:, :, :3]

        # Prepare the region of interest on the background
        roi = tf.image.crop_to_bounding_box(background, overlay_y, overlay_x, overlay_h, overlay_w)

        # Blend the images
        blended_region = overlay_rgb * alpha_mask + roi * (1 - alpha_mask)

        # Insert the blended region back into the background
        background_padded = tf.image.pad_to_bounding_box(blended_region, overlay_y, overlay_x, bg_h, bg_w)
        background_padded = tf.where(background_padded == 0, background, background_padded)

        return background_padded
    
    def call(self, inputs, training=True):
        if not training:
            return inputs
        
        if len(inputs.shape) == 3:  # Take single image
            inputs = tf.expand_dims(inputs, axis=0)

        def augment_image(img):
            if random.random() < self.augmentation_prob:
                return self.augment_single_image(img)
            else:
                return img
            
        augmented_images = tf.map_fn(augment_image, inputs)
            
        # augmented_images = tf.map_fn(lambda img: self.augment_single_image(img), inputs)

        if len(augmented_images.shape) == 4:  # If an extra dimension was added
            augmented_images = tf.squeeze(augmented_images, axis=0)  # Squeeze the extra dimension

        return augmented_images

# Load given sample image
def load_sample_image(image_path, image_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

# Visualize original and augmented image
def visualize_augmentation(original_image, augmented_image):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Augmented Image")
    plt.imshow(augmented_image)
    plt.axis('off')

    plt.show()

# ------------ EXAMPLE USAGE ------------ 
# # Example usage in a dataset map function
# def augment_image(image, label, augment_layer):
#     image = augment_layer(image, training=True)
#     return image, label

# # Example initialization
# augmentation_layer = RandomOcclusionLayer(
#     augmentation_prob=0.4,
#     sunglasses_path='face_recognition/datasets/augment/black_sunglasses.png',
#     hat_path='face_recognition/datasets/augment/hat.png',
#     mask_path='face_recognition/datasets/augment/mask.png'
# )

# ------------ Visualise for REPORT "Proposed Model" ------------ 
# # Paths
# image_path = 'face_recognition/datasets/celeb_a/split/train/10144/103673.jpg'
# sunglasses_path = 'face_recognition/datasets/augment/black_sunglasses.png'
# hat_path = 'face_recognition/datasets/augment/hat.png'
# mask_path = 'face_recognition/datasets/augment/mask.png'

# # Parameters
# image_size = (224, 224)

# # Load the original image
# original_image = load_sample_image(image_path, image_size)

# # Instantiate the augmentation layer
# augmentation_layer = RandomOcclusionLayer(
#     augmentation_prob=1.0,  # 40% chance of occlusion
#     sunglasses_path=sunglasses_path,
#     hat_path=hat_path,
#     mask_path=mask_path
# )

# # Apply the augmentation
# augmented_image = augmentation_layer(tf.expand_dims(original_image, axis=0), training=True)
# augmented_image = tf.squeeze(augmented_image, axis=0)

# # Visualize
# visualize_augmentation(original_image, augmented_image)