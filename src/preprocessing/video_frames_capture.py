import cv2
from mtcnn import MTCNN
import os

def video_to_frames(video_path, identity, dataset_dir="face_recognition/datasets/face_bank", scales=[1024, 1980], threshold=0.5):
    """Splits a video into frames, detects faces using MTCNN, and saves frames to the identity folder in the dataset."""

    detector = MTCNN()  # Use MTCNN face detector
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_count = 0
    save_count = 0

    identity_dir = os.path.join(dataset_dir, identity)
    os.makedirs(identity_dir, exist_ok=True)

    save_count = len([f for f in os.listdir(identity_dir) if (f.endswith('.jpg') or f.endswith('.jpeg'))])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)

        if faces:
            for face in faces:
                # x, y, width, height = face['box']
                # face_img = frame[y:y+height, x:x+width]  # Extract face region
                save_path = os.path.join(identity_dir, f"{save_count:03d}.jpg")
                cv2.imwrite(save_path, frame)
                save_count += 1

        frame_count += 1

    cap.release()
    print(f"Processed {frame_count} frames, saved {save_count} faces to {identity_dir}")

# Example Usage (assuming your 'face_bank' is in the current directory)
identity = "Nada"
video_path = f"face_recognition/datasets/face_videos/{identity}_video_4.mp4" 

video_to_frames(video_path, identity)