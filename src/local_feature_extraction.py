import numpy as np
import region_utils as region_utils
from mtcnn import MTCNN
from local_descriptors import LocalDescriptorExtractor
import cv2

class LocalFeatureExtractor:
    def __init__(self):
        self.detector = MTCNN()
        self.descriptor_extractor = LocalDescriptorExtractor()

    def fuse_regions(self, image, landmarks, sift=True, hog=False, lbp=False, region_norm=True):
        regions = region_utils.split_to_regions(image, landmarks)
        fused_features = np.array([])

        for region in regions:
            if region is None or region.size == 0:
                print("Error: Empty region for landmark!")
                continue
            region_features = self.descriptor_extractor.get_region_features(region, sift, hog, lbp)
            if region_norm:
                if np.linalg.norm(region_features, ord=2) == 0:
                    print("WARNING: Zero-norm region features encountered during region normalization")
                else:
                    region_features /= np.linalg.norm(region_features, ord=2)
            fused_features = np.concatenate([fused_features, region_features])

        if not region_norm:
            if np.linalg.norm(fused_features, ord=2) == 0:
                print("WARNING: Zero-norm vector encountered during global normalization")
            else:
                fused_features /= np.linalg.norm(fused_features, ord=2)
        
        print("Final grid descriptor shape: ", fused_features.shape)
        print(fused_features)
        return fused_features.flatten()
    
    def extract_local_features(self, image, identity=None):
        """ Extracts local features using RetinaFace. 
        
        Args:
            image: Image to detect faces and lanmdarks in. 
            
        Returns:
            Face data (landmarks, bounding box, features, etc.) on all detected faces. """
        
        # Detect faces and face data (landmarks, facial area) in given image
        faces = self.detector.detect_faces(image)  
        
        # Each face: preprocess -> extract local features -> store
        faces_features = {}
        for face in faces:
            landmarks = face['keypoints']
            bbox = face['box']

            face_img, new_landmarks = region_utils.preprocess_face(image, bbox, landmarks)   

            # Extract and store local features
            face_features = self.fuse_regions(face_img, new_landmarks)
            faces_features.setdefault(identity, []).extend(face_features)  

        # Return all faces with per-face extracted features
        return faces_features

    def get_and_store_features_from_dataset(self, dataset='face_recognition/datasets/face_bank'):
        images, identities = region_utils.load_dataset(dataset)
        features = {}
        for image, identity in zip(images, identities):
            if image is None:
                print("Image: ", image)
            print("ID: ", identity)
            feat_dict = self.extract_local_features(image, identity)
        
            # Check if a face was detected
            if feat_dict.get(identity):  # If a face was detected, this key will exist
                features.setdefault(identity, []).extend(feat_dict[identity])
            else:
                print(f"Skipping image for identity {identity} as no face was detected.")

        region_utils.save_features(features)

    def get_baseline_features_dict(self, dataset='face_recognition/datasets/face_bank'):
        images, identities = region_utils.load_dataset(dataset)
        features_dict = {}
        for image, identity in zip(images, identities):
            raw_features = self.descriptor_extractor.extract_raw_pixel_features(image)
            random_features = self.descriptor_extractor.extract_random_features(image)
            features_dict.setdefault(identity, []).extend([raw_features, random_features])
        region_utils.save_features(features_dict, filename="baseline_features.pkl")

if __name__ == "__main__":
    feature_extractor = LocalFeatureExtractor()

    # # --- Option 1: Extract and save features from the dataset ---
    # feature_extractor.get_and_store_features_from_dataset()
    # feature_extractor.get_baseline_features_dict()

    # # --- Option 2: Load features from a pre-saved file ---
    # print("loading")
    # features = region_utils.load_features()
    # print("loading finished")

    # # --- Calculate Distances ---
    # distances = region_utils.calculate_l2_dist_euclidian(features)
    # print("distances finished")

    # # --- Analyse and Visualize ---
    # region_utils.analyse_features(distances)
    # region_utils.histograms(distances)

    # # --- For Baseline Features ---
    # baseline_features = region_utils.load_features(filename="baseline_features.pkl")
    # raw_dist, random_dist = region_utils.calculate_l2_dist_with_baselines(baseline_features)
    # region_utils.analyse_features(raw_dist)
    # region_utils.histograms(raw_dist)
    # region_utils.analyse_features(random_dist)
    # region_utils.histograms(random_dist)

    # # Extract features from dataset and store
    # feature_extractor.get_and_store_features_from_dataset()

#     # # Load features and train classifier
#     # features_dict = local_utils.load_features()
#     # final_distances = local_utils.calculate_l2_dist_euclidian(features_dict)

#     # # Metrics and Distribution Analysis
#     # local_utils.analyse_features(final_distances)

#     # # Histograms of inter- and intra- distances
#     # local_utils.histograms(final_distances)

    # Visualise Preprocessing Results
    image = cv2.imread('face_recognition/datasets/face_bank/Nada/033.jpeg')

    # Detect Faces
    faces = feature_extractor.detector.detect_faces(image)

    for face in faces:
        landmarks = face['keypoints']
        bbox = face['box']
        face_img, resized_landmarks = region_utils.preprocess_face(image.copy(), bbox, landmarks)

        print("Resized Image Shape: ", face_img.shape)
        for landmark_name, (x, y) in resized_landmarks.items():
            print(f"New Landmark: {landmark_name}, Position: ({x}, {y})")  

        # # Visualise landmark-centric regions
        # regions = region_utils.split_to_regions(face_img, resized_landmarks)

        # # Visualise grid
        grid_size = (2, 2)
        regions = region_utils.split_into_grid(face_img, grid_size)

        # Display the images with landmarks
        region_utils.visualize_preprocessed_face(face_img, resized_landmarks) 

        visualized_image = region_utils.visualize_regions(face_img, regions, resized_landmarks, grid=True, grid_size=grid_size)
        cv2.imshow("Face Regions", visualized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()