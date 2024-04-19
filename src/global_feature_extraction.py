import cv2
import numpy as np
from tensorflow import keras
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model

def load_pretrained_model():
    base_model = ResNet50(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

def align_face(image, landmarks):
    # Define the desired eye positions
    desired_left_eye=(0.35, 0.35)
    desired_face_width = 224
    desired_face_height = 224

    # Extract the left and right eye (x, y)-coordinates
    leftEye = landmarks['left_eye']
    rightEye = landmarks['right_eye']

    # Compute the angle between the eye centroids
    dY = rightEye[1] - leftEye[1]
    dX = rightEye[0] - leftEye[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # Compute the desired right eye x-coordinate based on the desired x-coordinate
    desired_right_eye_x = 1.0 - desired_left_eye[0]

    # Determine the scale of the new resulting distance between the eyes
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_right_eye_x - desired_left_eye[0])
    desired_dist *= desired_face_width
    scale = desired_dist / dist

    # Compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEye[0] + rightEye[0]) // 2, (leftEye[1] + rightEye[1]) // 2)

    # Grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # Update the translation component of the matrix
    tX = desired_face_width * 0.5
    tY = desired_face_height * desired_left_eye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # Apply the affine transformation
    (w, h) = (desired_face_width, desired_face_height)
    output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    return output

def extract_faces(img, detection):
    faces = []
    for face, data in detection.items():
        if data['score'] > 0.5:  # Confidence threshold of 0.5, can be later adjusted
            x1, y1, x2, y2 = data['facial_area']
            face = img[y1:y2, x1:x2]
            faces.append(face)
    return faces

def preprocess_face(face, target_size=(224, 224)):
    # TODO: check if normalisation process is correct
    face = cv2.resize(face, target_size)
    face_img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_img = image.img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    face_img = preprocess_input(face_img)
    return face_img

def extract_global_features(face_img, model):
    # TODO: pre-trained model can be fine-tuned on my own data with synthetic occlusions
    features = model.predict(face_img)
    return features

# Main execution
model = load_pretrained_model()
image_path = 'path_to_image.jpg'
detections = [{'box': [100, 100, 50, 50], 'confidence': 0.9}, {'box': [200, 200, 50, 50], 'confidence': 0.92}]

faces = extract_faces(image_path, detections)
for face in faces:
    preprocessed_face = preprocess_face(face)
    features = extract_global_features(preprocessed_face, model)
    print(features.shape)  # Example of how features might look
