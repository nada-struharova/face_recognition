import cv2
from mtcnn import MTCNN
from region_utils import preprocess_face
import matplotlib.pyplot as plt

# Load the image
image_path = 'face_recognition/datasets/IMG_4785.jpeg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize the MTCNN detector
detector = MTCNN()

# Detect faces in the image
detections = detector.detect_faces(image_rgb)

# Process each detected face
for i, detection in enumerate(detections):
    bbox = detection['box']
    keypoints = detection['keypoints']
    
    # Preprocess the face
    face_img, new_landmarks = preprocess_face(image_rgb, bbox, keypoints)

    # Visualize the preprocessed face
    plt.figure(figsize=(10, 6))
    
    # Original image with bounding boxes and landmarks
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image (Bounding Box, Landmarks)')
    plt.axis('off')
    
    # Preprocessed face with adjusted landmarks
    plt.subplot(1, 2, 2)
    plt.imshow(face_img)
    for landmark, (x, y) in new_landmarks.items():
        plt.scatter(x, y, c='g', s=40)
        plt.text(x + 5, y - 5, landmark, fontsize=12, color='white')  # Adjust text position
    plt.title('Preprocessed Face {}'.format(i+1))
    plt.axis('off')
    
    plt.show()