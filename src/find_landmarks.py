import os
import cv2
import csv
import numpy as np
from mtcnn import MTCNN

def adjust_landmarks(original_landmarks, bbox, original_image_shape, target_image_shape=(224, 224)):
    """
    Adjust landmarks based on transformations applied to the image (cropped and resized).

    Args:
    - original_landmarks: Dictionary containing original landmark coordinates.
    - bbox: Bounding box coordinates (x, y, width, height).
    - original_image_shape: Tuple containing the shape of the original image (height, width).
    - target_image_shape: Tuple indicating the target image shape after resizing (default is (224, 224)).

    Returns:
    - Adjusted landmark coordinates as a dictionary.
    """
    x, y, width, height = bbox
    orig_height, orig_width = original_image_shape
    target_height, target_width = target_image_shape
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height
    adjusted_landmarks = {}

    for landmark, (landmark_x, landmark_y) in original_landmarks.items():
        # Adjust landmark coordinates based on cropping and resizing
        adjusted_x = (landmark_x - x) * (target_width / width)
        adjusted_y = (landmark_y - y) * (target_height / height)
        adjusted_landmarks[landmark] = (adjusted_x, adjusted_y)

    return adjusted_landmarks

def detect_landmarks(image_path):
    """Detects facial landmarks using MTCNN detector.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary of facial landmarks adjusted for cropping and resizing.
    """
    detector = MTCNN()
    
    image = cv2.imread(image_path)
    original_shape = (image.shape[0], image.shape[1])
    faces = detector.detect_faces(image)

    if faces:
        face = faces[0]
        bbox = face['box']
        landmarks = face['keypoints']
        adjusted_landmarks = adjust_landmarks(landmarks, bbox, original_shape)
        return adjusted_landmarks
    else:
        print(f"No face detected in {image_path}")
        return None

def save_landmarks(dataset_dir, output_file):
    """Finds and saves facial landmarks for images in the dataset directory.

    Args:
        dataset_dir: Directory containing the image files.
        output_file: Path to the output CSV file to save the landmark data.
    """
    # Read the already processed filenames
    processed_files = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                processed_files.add(row[0])

    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not processed_files:
            writer.writerow(['Image', 'Landmark', 'X', 'Y'])

        for filename in os.listdir(dataset_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                if filename in processed_files:
                    continue

                image_path = os.path.join(dataset_dir, filename)
                landmarks = detect_landmarks(image_path)
                if landmarks:
                    for landmark, (x, y) in landmarks.items():
                        writer.writerow([filename, landmark, x, y])
                    print(f"Landmarks saved for {filename}")

import pandas as pd

# # Example usage:
# dataset_dir = 'face_recognition/datasets/celeb_a/img_align_celeba'
output_file = 'face_recognition/datasets/celeb_a/detected_cropped_landmarks.csv'
# save_landmarks(dataset_dir, output_file)

# Load landmarks from CSV
df = pd.read_csv(output_file)

# Calculate mean coordinates for each landmark
average_landmarks = df.groupby('Landmark')[['X', 'Y']].mean().round(2).to_dict('index')

print(average_landmarks)