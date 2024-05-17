import numpy as np
import utils
import descriptors
from retinaface import RetinaFace
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2

# ------------------ Region Based Features ------------------ 
def get_region_features(region, sift, surf, hog, lbp, daisy=False, orb=False):
    """ Extract local features of region.
    Args:
        region: Specific regions for local feature extraction.
        sift: Include SIFT descriptor in final concatenation.
        surf: Include SURF descriptor in final concatenation.
        hog: Include HOG descriptor in final concatenation.
        
    Returns:
        Concatenated local feature descriptors for given region. """
    
    features = np.array([])

    if sift:
        sift_feat = descriptors.extract_sift_features(region)
        if not isinstance(sift_feat, np.ndarray):
            sift_feat = np.array(sift_feat)  # Convert to np array
        features = np.concatenate([features, sift_feat])  # Concatenate NumPy arrays
    if surf:
        surf_feat= descriptors.extract_surf_features(region)
        if not isinstance(surf_feat, np.ndarray):
            surf_feat = np.array(surf_feat)  # Convert to np array
        features = np.concatenate([features, surf_feat])
    if hog:
        hog_feat = descriptors.extract_hog_features(region)
        if not isinstance(hog_feat, np.ndarray):
            hog_feat = np.array(hog_feat)  # Convert to np array
        features = np.concatenate([features, hog_feat])
    if lbp:
        lbp_feat = descriptors.extract_lbp_features(region)
        if not isinstance(lbp_feat, np.ndarray):
            lbp_feat = np.array(lbp_feat)  # Convert to np array
        features = np.concatenate([features, lbp_feat])
    if daisy:
        daisy_feat = descriptors.extract_daisy_descriptors(region)
        daisy_feat = daisy_feat.flatten()


    return features

def fuse_regions(image, facial_area, landmarks, sift=True, surf=False, hog=False, lbp=False, region_norm=True):
    """ Extract all wanted local features. """

    # Split into regions (grid-based or landmark-centric)
    regions = utils.split_into_landmark_regions(image, landmarks)
    fused_features = np.array([])

    ### if using split_into_regions_landmarks use "for region, landmark in regions" ###
    for region in regions:
        # Error Handling: Empty region / grid cell
        if region is None or region.size == 0:
            print("Error: Empty region for landmark!")
            continue 

        # Extract Local Features
        region_features = get_region_features(region, sift, surf, hog, lbp)

        # Individual normalisation (L2 norm)
        if region_norm:
            if np.linalg.norm(region_features, ord=2) == 0:  # Check for zero-norm
                print("WARNING: Zero-norm region features encountered during region normalization")
                # skip normalisation
            else:
                region_features /= np.linalg.norm(region_features, ord=2)


        # Concatenate directly
        fused_features = np.concatenate([fused_features, region_features])        

    # Global Normalisation (L2 norm)
    if not region_norm:
        if np.linalg.norm(fused_features, ord=2) == 0: 
            print("WARNING: Zero-norm vector encountered during global normalization")
            # skip normalisation in this case
        else:
            fused_features /= np.linalg.norm(fused_features, ord=2)

    print("Final grid descriptor shape: ", fused_features.shape)
    return fused_features.flatten()

# ------------------ Feature Extraction ------------------ 
def extract_local_features(image, identity=None):
    """ Extracts local features using RetinaFace. 
    
    Args:
        image: Image to detect faces and lanmdarks in. 
        
    Returns:
        Face data (landmarks, bounding box, features, etc.) on all detected faces. """
    
    # Detect faces and face data (landmarks, facial area) in given image
    faces = RetinaFace.detect_faces(image)  
    
    # Each face: preprocess -> extract local features -> store
    faces_features = {}
    for face_id, face_data in faces.items():
        face_img, landmarks = utils.preprocess_face(image, face_data)   

        # Extract and store local features
        face_features = fuse_regions(face_img, face_data['facial_area'], landmarks)
        faces_features.setdefault(identity, []).extend(face_features)  

    # Return all faces with per-face extracted features
    return faces_features

def get_and_store_features_from_dataset(dataset='face_recognition/datasets/evaluate_local'):
    images, identities = utils.load_dataset(dataset) 
    features = {} 

    for image, identity in zip(images, identities):
        # Debugging and Error Handling
        if image is None:
            print("Image: ", image)
        print("ID: ", identity)
        feat_dict = extract_local_features(image, identity)
        features.setdefault(identity, []).extend(feat_dict[identity])
    
    utils.save_features(features)

def get_baseline_features_dict(dataset='face_recognition/datasets/evaluate_local'):
    images, identities = utils.load_dataset(dataset)
    features_dict = {}

    for image, identity in zip(images, identities):
        raw_features = descriptors.extract_raw_pixel_features(image)
        random_features = descriptors.extract_random_features(image)

        # Store both types of features for analysis 
        features_dict.setdefault(identity, []).extend([raw_features, random_features])

    utils.save_features(features_dict, filename="baseline_features.pkl")

# ------------------ Random Forest Classifier Train + Evaluate ------------------

def train_and_eval_random_forest(local_features):
    """Train a Random Forest classifier on local features and evaluate performance.
    
    Args:
        features: A dictionary where keys are identities and values are lists of local features.

    Returns:
        Accuracy of the Random Forest classifier on the test set.
    """

    # 1. Data Preparation
    all_features = []
    all_labels = []

    for identity, feature_list in local_features.items():
        for face_features in feature_list:
            all_features.append(face_features)
            all_labels.append(identity)

    X = np.array(all_features)
    y = np.array(all_labels)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42) 
    clf.fit(X_train, y_train)

    # 4. Prediction and Evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of Random Forest on local features: {accuracy:.2f}")

    return accuracy, clf  # Return both accuracy and the trained classifier

# # ------------------ Program ------------------
# get_and_store_features_from_dataset()

# features_dict = utils.load_features()

# # Calculate intra- and inter-identity distances of features
# final_distances = utils.calculate_l2_dist_euclidian(features_dict)

# # ------------------ Analysis & Visualisation ------------------ 

# # Metrics and Distribution Analysis
# utils.analyse_features(final_distances)

# # Histograms of inter- and intra- distances
# utils.histograms(final_distances)

# # ------------------ Random Forest Classifier Training ------------------ 
# accuracy, trained_rf_classifier = train_and_eval_random_forest(features_dict)

# ------------------ Visualise Preprocessing Results ------------------ 
image = cv2.imread('face_recognition/datasets/evaluate_local/Nada/025.jpeg')  
faces = RetinaFace.detect_faces(image)  

for face_id, face_data in faces.items():
    face_img, resized_landmarks = utils.preprocess_face(image.copy(), face_data)

    print("Resized Image Shape: ", face_img.shape)
    for landmark_name, (x, y) in resized_landmarks.items():
        print(f"New Landmark: {landmark_name}, Position: ({x}, {y})")  

    # Display the images with landmarks
    utils.visualize_preprocessed_face(face_img, resized_landmarks) 

    rois = utils.extract_roi_around_landmarks(face_img, resized_landmarks)
    utils.visualise_rois(face_img, resized_landmarks, rois)