import os
import cv2
from mtcnn import MTCNN
import tensorflow as tf

# Initialize MTCNN detector
detector = MTCNN()

# Define paths
celeba_dir = 'face_recognition/datasets/celeb_a/img_align_celeba'
cropped_dir = 'face_recognition/datasets/celeb_a/img_align_celeba_cropped'

def is_valid_image(file_path):
    """
    Checks if an image file can be loaded successfully.

    Args:
        file_path (str): Path to the image file.

    Returns:
        bool: True if the image is valid, False otherwise.
    """
    try:
        image = tf.io.read_file(file_path)
        tf.image.decode_jpeg(image)  
        return True
    except (tf.errors.InvalidArgumentError, OSError):
        return False

def check_images_in_directory(image_dir):
    """
    Scans a directory and checks if any images are invalid (None when loaded).

    Args:
        image_dir (str): Path to the directory containing images.
    """
    invalid_images = []

    for filename in os.listdir(image_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):  # Adjust image extensions as needed
            file_path = os.path.join(image_dir, filename)
            if not is_valid_image(file_path):
                invalid_images.append(file_path)

    if invalid_images:
        print("The following images are invalid (could not be loaded):")
        for img in invalid_images:
            print(img)
    else:
        print("All images in the directory are valid.")

# Example usage
image_directory = 'face_recognition/datasets/celeb_a/img_align_celeba_cropped'  # Replace with your actual directory
check_images_in_directory(image_directory)

# Function to detect and crop faces
def detect_and_crop_face(image_path, save_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    detections = detector.detect_faces(img_rgb)

    if detections:
        # Assume the largest face is the main face
        largest_face = max(detections, key=lambda det: det['box'][2] * det['box'][3])
        x, y, w, h = largest_face['box']
        cropped_img = img[y:y+h, x:x+w]
        cv2.imwrite(save_path, cropped_img)
    else:
        # If no face is detected, save the original image
        print("No face detected in image ", image_path)
        cv2.imwrite(save_path, img)

# # Iterate through images in celeba directory
# for filename in os.listdir(celeba_dir):
#     celeba_path = os.path.join(celeba_dir, filename)
#     cropped_path = os.path.join(cropped_dir, filename)

#     # Check if the image is not already cropped
#     if not os.path.exists(cropped_path):
#         # Detect and crop faces
#         detect_and_crop_face(celeba_path, cropped_path)
