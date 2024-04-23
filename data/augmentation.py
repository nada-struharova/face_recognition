import cv2
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import preprocess_input
import random
import matplotlib.pyplot as plt

def synthetic_occlusion(image, occlusion_probability=0.3):
    if random.random() < occlusion_probability:
        h, w, _ = image.shape
        # Define occlusion size and position randomly
        occ_height = random.randint(h // 4, h // 2)
        occ_width = random.randint(w // 4, w // 2)
        occ_x = random.randint(0, w - occ_width)
        occ_y = random.randint(0, h - occ_height)

        # Apply occlusion
        image[occ_y:occ_y + occ_height, occ_x:occ_x + occ_width] = 0
    return image

def occlude_rectangle(image, severity=0.3):
    """Adds a rectangle occlusion to the image.
    Args:
        image: Input image as a NumPy array (OpenCV format).
        severity: Float between 0.0 and 1.0. Controls size of occlusion.
    """
    height, width = image.shape[:2]

    # Generate
    occ_width = int(width * severity)
    occ_height = int(height * severity)
    x_start = np.random.randint(0, width - occ_width)
    y_start = np.random.randint(0, height - occ_height)

    color = tuple(np.random.randint(256, size=3))  # Random color for occlusion
    cv2.rectangle(image, (x_start, y_start), (x_start + occ_width, y_start + occ_height), color, -1)

    return image

def occlude_ellipse(image, severity=0.2):
    """Adds an elliptical occlusion to the image.
    Args:
        image: Input image as a NumPy array (OpenCV format).
        severity: Float between 0.0 and 1.0. Controls size of occlusion.
    """
    height, width = image.shape[:2]

    # Random parameters based on severity
    center_x = np.random.randint(0, width)
    center_y = np.random.randint(0, height)
    max_radius_y = int(height * severity)
    max_radius_x = int(width * severity)
    radius_y = np.random.randint(1, max_radius_y)
    radius_x = np.random.randint(1, max_radius_x)
    angle = np.random.randint(0, 360)

    # Create ellipse mask
    mask = np.zeros((height, width), dtype=np.uint8)  # Black background
    cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), angle, 0, 360, (255, 255, 255), -1)

    # Apply mask to image (occlusion)
    occluded_image = cv2.bitwise_and(image, image, mask=mask)

    return occluded_image

def add_occlusion(image):
    """ Add either a rectangle or ellipse occlusion randomly """
    if np.random.rand() > 0.5:
        return occlude_rectangle(image)
    else:
        return occlude_ellipse(image)


def load_and_preprocess_image(image_path, target_size=(224, 224), apply_occlusion=False):
    # Load image
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)

    # Optionally apply synthetic occlusion
    if apply_occlusion:
        image = synthetic_occlusion(image)

    # Normalize image
    image = preprocess_input(image)  # Use appropriate preprocessing based on the model you plan to use
    return image

def load_dataset(directory, num_images=1000, occlusion_probability=0.3):
    images = []
    """ Apply occlusions to approximately 30% of the images. """
    for img_file in os.listdir(directory)[:num_images]:
        img_path = os.path.join(directory, img_file)
        to_occlude = random.random() < occlusion_probability
        try:
            img = load_and_preprocess_image(img_path, to_occlude=to_occlude)
            images.append(img)
        except Exception as e:
            print(f"Failed to process image {img_file}: {e}")
    return np.array(images)

def show_images(original, occluded):
    """ Testing synthethic occlusion. """
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(occluded, cv2.COLOR_BGR2RGB))
    plt.title('Occluded Image')
    plt.axis('off')
    plt.show()

# Load an image (update the path to your image file)
image_path = 'face-alignment/test/assets/aflw-test.jpg'
image = cv2.imread(image_path)

# Augment with occlusions
occluded_with_rectangle = occlude_rectangle(image)
occluded_with_ellipse = occlude_ellipse(image)

# Display images
show_images(image, occluded_with_rectangle)
show_images(image, occluded_with_ellipse)

