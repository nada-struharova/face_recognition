import os
import cv2
import numpy as np
from local_feature_extraction import LocalFeatureExtractor
from tqdm import tqdm  # Progress bar

def extract_and_save_local_features(image_dir, save_path="local_features.npz", batch_size=250):
    """Extracts and saves local SIFT features for images in batches."""

    local_feature_extractor = LocalFeatureExtractor()

    avg_landmarks = {'left_eye': (63, 85),
                              'mouth_left': (68, 170),
                              'mouth_right': (159, 171),
                              'nose': (113, 132),
                              'right_eye': (167, 85)}
        
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

    # Create the NPZ file in write mode initially
    with open(save_path, 'wb') as f:
        pass  # This just creates the file if it doesn't exist

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in batch_paths]

        for image_path, image in zip(batch_paths, batch_images):
            features = local_feature_extractor.fuse_regions(image, avg_landmarks)

            if features.size > 0:
                # Append the batch features to the NPZ file
                with open(save_path, 'ab') as f:  # Open in append binary mode ('ab')
                    np.savez(f, {image_path: features})  # Save the batch features
            else:
                print(f"Warning: No features for {image_path}. Skipping...")
                continue

if __name__ == "__main__":
    # Example usage:
    image_dir = "face_recognition/datasets/celeb_a/img_align_celeba" 
    save_path = "local_features.npz"
    extract_and_save_local_features(image_dir, save_path)
