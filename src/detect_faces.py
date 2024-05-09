from retinaface import RetinaFace
import cv2  # Might be needed for image loading
import matplotlib.pyplot as plt

def get_face_data(image_path):
    resp = RetinaFace.detect_faces(image_path)

    if len(resp) > 0:  # Check if at least one face is detected
        face_data = resp['face_1']  # Access the first detected face

        # Directly access and return the landmarks
        return face_data
    else:
        return None  # Return None if no face is found

# Example Usage
image_path = 'face_recognition/datasets/test_images/single_face.png'  # Replace with the path to your image
face_data = get_face_data(image_path)
print(face_data)

# faces = RetinaFace.extract_faces(img_path = "face_recognition/datasets/test_images/single_face.png", align = True)
# for face in faces:
#   plt.imshow(face)
#   plt.show()