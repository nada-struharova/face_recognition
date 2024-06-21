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
            'sunglasses': 1.1,
            'hat': 1.0,
            'mask': 2.0
        }
        self.avg_landmarks = {'left_eye': (63, 85),
                              'mouth_left': (68, 170),
                              'mouth_right': (159, 171),
                              'nose': (113, 132),
                              'right_eye': (167, 85)}

    def load_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=4)
        image = tf.image.resize(image, [224, 224])
        return image
    
    def augment_single_image(self, image):
        # Select a random augmentation method
        aug_img_type = random.choice(list(self.augmentation_images.keys()))
        aug_img = self.augmentation_images[aug_img_type]

        # Scale based on selected augmentation
        scaling_factor = self.scaling_factors[aug_img_type]  # Use fixed scaling factors
        
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
        
        overlay_x, overlay_y = overlay_position
        # Calculate cropping parameters 
        crop_top = int(max(0, -overlay_y))  
        crop_left = int(max(0, -overlay_x))  
        crop_bottom = int(max(0, overlay_y + overlay_h - 224)) 
        crop_right = int(max(0, overlay_x + overlay_w - 224))  

        # Crop to (224, 224) before applying
        aug_img_resized = tf.image.crop_to_bounding_box(
            aug_img_resized,
            crop_top,
            crop_left,
            overlay_h - crop_top - crop_bottom,  
            overlay_w - crop_left - crop_right 
        )
        overlay_position_x = int(max(0, overlay_x))
        overlay_position_y = int(max(0, overlay_y))

        image = self.apply_overlay(image, aug_img_resized, (max(0, overlay_position_x), max(0, overlay_position_y)))  # Adjusted position

        if aug_img_type == 'sunglasses':
            overlay_position = self.get_sunglasses_position(self.avg_landmarks, overlay_w, overlay_h)
        elif aug_img_type == 'hat':
            overlay_position = self.get_hat_position(self.avg_landmarks, overlay_w, overlay_h)
        elif aug_img_type == 'mask':
            overlay_position = self.get_mask_position(self.avg_landmarks, overlay_w, overlay_h)
        
        image = self.apply_overlay(image, aug_img_resized, overlay_position)

        # Geometric Transformations
        if random.random() < 0.2:
            image = tf.image.random_crop(image, size=[200, 200, 3])  
            image = tf.image.resize(image, (224, 224)) 
        if random.random() < 0.5:
            image = tf.image.random_flip_left_right(image) 
        if random.random() < 0.5:
            image = tf.image.rot90(image, k=random.randint(1, 3))

        # Color Transformations
        if random.random() < 0.5:
            image = tf.image.random_brightness(image, max_delta=0.2)
        if random.random() < 0.5:
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2) 
        if random.random() < 0.5:
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2) 

        return image
    
    def get_sunglasses_position(self, avg_landmarks, overlay_w, overlay_h):
        # for cropped image landmarks
        center_x = (avg_landmarks['left_eye'][0] + avg_landmarks['right_eye'][0]) // 2
        center_y = (avg_landmarks['left_eye'][1] + avg_landmarks['right_eye'][1]) // 2
        position_x = center_x - overlay_w // 2
        position_y = center_y - overlay_h // 2 - 10 
        return (position_x, position_y)

    def get_hat_position(self, avg_landmarks, overlay_w, overlay_h):
        # for cropped image landmarks and smaller hat
        center_x = (avg_landmarks['left_eye'][0] + avg_landmarks['right_eye'][0]) // 2
        position_x = center_x - overlay_w // 2 
        position_y = avg_landmarks['left_eye'][1] - overlay_h // 1.2 
        return (position_x, position_y)

    def get_mask_position(self, avg_landmarks, overlay_w, overlay_h):
        # for cropped image landmarks and smaller mask
        center_x = avg_landmarks['nose'][0]
        center_y = (avg_landmarks['nose'][1] + avg_landmarks['mouth_left'][1]) // 2
        position_x = center_x - overlay_w // 2
        position_y = center_y - overlay_h // 2 + 10  
        return (position_x, position_y)

    def apply_overlay(self, background, overlay_image, position):
        bg_h, bg_w, _ = background.shape
        overlay_h, overlay_w, _ = overlay_image.shape
        
        # Adjust x, y position based on resized overlay image
        overlay_x = int(position[0])
        overlay_y = int(position[1])

        # Ensure overlay is within bounds
        overlay_x = max(0, min(bg_w - overlay_w, overlay_x))
        overlay_y = max(0, min(bg_h - overlay_h, overlay_y))

        # Adjust overlay width
        overlay_w = min(overlay_w, bg_w - overlay_x)

        # Extract alpha channel (to apply overlay)
        alpha_mask = overlay_image[:, :, 3] / 255.0
        alpha_mask = tf.stack([alpha_mask] * 3, axis=-1)  # Create 3-channel alpha mask

        # Extract RGB channels
        overlay_rgb = overlay_image[:, :, :3]

        # Get ROI on background
        roi = tf.image.crop_to_bounding_box(background, overlay_y, overlay_x, overlay_h, overlay_w)
        blended_region = overlay_rgb * alpha_mask + roi * (1 - alpha_mask)

        # Insert blended region back into the background
        background_padded = tf.image.pad_to_bounding_box(blended_region, overlay_y, overlay_x, bg_h, bg_w)
        background_padded = tf.where(background_padded == 0, background, background_padded)

        return background_padded
    
    def call(self, inputs, training=True):
        if not training:
            return inputs
        
        if len(inputs.shape) == 3:  # Take single image from batch
            inputs = tf.expand_dims(inputs, axis=0)

        def augment_image(img):
            if random.random() < self.augmentation_prob:
                return self.augment_single_image(img)
            else:
                return img
            
        augmented_images = tf.map_fn(augment_image, inputs)
            
        # augmented_images = tf.map_fn(lambda img: self.augment_single_image(img), inputs)

        if len(augmented_images.shape) == 4:
            augmented_images = tf.squeeze(augmented_images, axis=0)

        return augmented_images

# Load given sample image
def load_sample_image(image_path, image_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

def visualize_augmentation(original_image, augmented_image):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image / 255.0)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Augmented Image")
    plt.imshow(augmented_image / 255.0)
    plt.axis('off')

    plt.show()

# ------------ Visualise for REPORT "Proposed Model" ------------ 
# Paths
image_path = 'face_recognition/datasets/celeb_a/split_fr_cropped/test/1/032404.jpg'
sunglasses_path = 'face_recognition/datasets/augment/black_sunglasses.png'
hat_path = 'face_recognition/datasets/augment/hat.png'
mask_path = 'face_recognition/datasets/augment/mask.png'

image_size = (224, 224)

# Load the original image
original_image = load_sample_image(image_path, image_size)

# Instantiate the augmentation layer
augmentation_layer = RandomOcclusionLayer(
    augmentation_prob=1.0, 
    sunglasses_path=sunglasses_path,
    hat_path=hat_path,
    mask_path=mask_path
)

# Apply the augmentation (no need to expand dims here as call function adds the batch dimension)
augmented_image = augmentation_layer(original_image, training=True)  

# Remove the batch dimension if it exists
if len(augmented_image.shape) == 4:
    augmented_image = tf.squeeze(augmented_image, axis=0)  
    
# Visualize
visualize_augmentation(original_image, augmented_image) 