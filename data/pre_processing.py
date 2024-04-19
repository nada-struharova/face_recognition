import cv2
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import preprocess_input
import random

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
    # Apply occlusions to approximately 30% of the images
    for img_file in os.listdir(directory)[:num_images]:
        img_path = os.path.join(directory, img_file)
        to_occlude = random.random() < occlusion_probability
        try:
            img = load_and_preprocess_image(img_path, to_occlude=to_occlude)
            images.append(img)
        except Exception as e:
            print(f"Failed to process image {img_file}: {e}")
    return np.array(images)

