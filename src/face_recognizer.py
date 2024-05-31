import numpy as np
import os
import cv2
import tensorflow as tf
import joblib
import argparse
from sklearn.preprocessing import StandardScaler
from retinaface import RetinaFace
from src.utils import *
import time

class Recognizer:
    def __init__(self, vgg16_model_path, fine_tuned_model_path, scaler_path):
        self.vgg16_model = self.load_vgg16_model(vgg16_model_path)
        self.fine_tuned_model = tf.keras.models.load_model(fine_tuned_model_path)
        self.scaler = joblib.load(scaler_path)
        self.face_bank = None
        self.names = []

    def load_vgg16_model(self, weights_path):
        base_model = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
        model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
        model.load_weights(weights_path, by_name=True)
        return model

    def extract_global_features(self, image):
        img = cv2.resize(image, (224, 224))
        img = tf.keras.applications.vgg16.preprocess_input(img)
        features = self.vgg16_model.predict(np.expand_dims(img, axis=0))
        return features.flatten()

    def extract_local_features(self, image):
        faces = RetinaFace.detect_faces(image)
        local_features = []

        for face_id, face_data in faces.items():
            face_img, landmarks = preprocess_face(image, face_data)
            face_features = fuse_regions(face_img, face_data['facial_area'], landmarks)
            local_features.extend(face_features)

        return np.array(local_features)

    def combine_features(self, global_features, local_features):
        return np.concatenate([global_features, local_features])

    def update_face_bank(self, face_bank_path='./face_bank'):
        embeddings = []
        names = []

        for person_name in os.listdir(face_bank_path):
            person_dir = os.path.join(face_bank_path, person_name)
            if os.path.isdir(person_dir):
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    image = cv2.imread(image_path)
                    if image is not None:
                        global_features = self.extract_global_features(image)
                        local_features = self.extract_local_features(image)
                        combined_features = self.combine_features(global_features, local_features)
                        embeddings.append(combined_features)
                        names.append(person_name)

        embeddings = self.scaler.fit_transform(embeddings)
        self.face_bank = np.array(embeddings)
        self.names = names
        joblib.dump(self.face_bank, 'face_bank_embeddings.pkl')
        joblib.dump(self.names, 'face_bank_names.pkl')

    def load_face_bank(self):
        self.face_bank = joblib.load('face_bank_embeddings.pkl')
        self.names = joblib.load('face_bank_names.pkl')

    def recognize(self, image):
        # Extract global and local features
        global_features = self.extract_global_features(image)
        local_features = self.extract_local_features(image)
        
        # Combine features
        combined_features = self.combine_features(global_features, local_features)
        
        # Normalize features
        combined_features = self.scaler.transform([combined_features])
        
        # Calculate distances
        dists = np.linalg.norm(self.face_bank - combined_features, axis=1)
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        
        # Thresholding
        threshold = 1.0  # Adjust the threshold based on your use-case
        if min_dist < threshold:
            return self.names[min_dist_idx]
        else:
            return 'Unknown'

def run_webcam_recognition(recognizer, args):
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    print('input video fps:', cap_fps)

    if not args.origin_size:
        width = width // 2
        height = height // 2

    if args.save:
        video_writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_file_path = os.path.join(args.output, 'webcam_output.mp4')
        video_writer = cv2.VideoWriter(output_file_path, video_writer_fourcc, cap_fps, (width, height))

    while cap.isOpened():
        tic = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        if not args.origin_size:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        predicted_identity = recognizer.recognize(frame_rgb)
        
        for face_id, face_data in RetinaFace.detect_faces(frame_rgb).items():
            bbox = face_data['facial_area']
            frame = draw_box_name(frame, bbox.astype("int"), predicted_identity)

        toc = time.time()
        real_fps = round(1 / (toc - tic), 4)
        frame = cv2.putText(frame, f"fps: {real_fps}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)

        if args.show:
            cv2.imshow('face Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if args.save:
            video_writer.write(frame)

    cap.release()
    if args.save:
        video_writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print('finish!')

def draw_box_name(frame, bbox, name):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame

def preprocess_face(image, face_data):
    facial_area = face_data['facial_area']
    landmarks = face_data['landmarks']
    face_img = image[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
    return face_img, landmarks

def load_names():
    # Implement loading of identity names, e.g., from a file
    return ["Unknown", "Person1", "Person2", "Person3"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Recognition - VGG16 with Local Features')
    parser.add_argument("--output", default="io/output", type=str, help="output dir path")
    parser.add_argument("--origin-size", default=True, action="store_true", help='Whether to use origin image size to evaluate')
    parser.add_argument("--show", default=True, action="store_true", help="show result")
    parser.add_argument("--save", default=False, action="store_true", help="whether to save")
    parser.add_argument("--update", default=False, action="store_true", help="whether to update the face bank")
    args = parser.parse_args()

    recognizer = Recognizer(
        vgg16_model_path='path_to_vgg16_weights.h5',
        fine_tuned_model_path='path_to_fine_tuned_model.h5',
        scaler_path='path_to_scaler.pkl'
    )

    if args.update:
        recognizer.update_face_bank()
        print('face bank updated')
    else:
        recognizer.load_face_bank()
        print('face bank loaded')

    run_webcam_recognition(recognizer, args)

