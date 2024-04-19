from skimage import io, gabor, img_as_float
import cv2
import numpy as np
import utils

def get_region(image, center, size=20):
    """ Extract a square region around a given center in an image. 
        - segment face into regions around five facial landmarks (right eye, left eye, 
          nose, right mouth corner, left mouth corner) 
        - 5 landmarks provided from RetinaFace pre-trained face detection model 
        - size=40 is arbitrary, adjust based on expected size of facial features """
    x, y = int(center[0]), int(center[1])
    x1, y1, x2, y2 = x - size, y - size, x + size, y + size
    if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        return None  # Region is out of image bounds
    return image[y1:y2, x1:x2]

def extract_surf_features(image, hessianThreshold=400):
    """ Extract SURF features from an face region. """
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
    _, descriptors = surf.detectAndCompute(image, None)
    if descriptors is None:
        descriptors = np.zeros((1, surf.descriptorSize()), dtype=np.float32)
    return descriptors.flatten()

def extract_hog_features(image, win=(64, 64), block=(16,16), stride=(8,8), cell=(8,8), bins=9):
    """ Extract HOG features from an face region. """
    hog = cv2.HOGDescriptor(win, block, stride, cell, bins)
    hog_features = hog.compute(image)
    return hog_features.flatten()

def extract_features(region):
    features = []

    surf_features = extract_surf_features(region)
    hog_features = extract_hog_features(region)

    # normalise features
    surf_norm = utils.normalise_features(surf_features)
    hog_norm = utils.normalise_features(hog_features)

    # concatenate local features extracted from regions (for per-face descriptor)
    features.append(np.concatenate((surf_norm, hog_norm)))    

def concatenate_features(features_dict):
    """ Concatenate features from all regions to form a complete face descriptor. """
    all_features = []
    for features in features_dict.values():
        concatenated_features = np.concatenate((features['surf'], features['hog']))
        all_features.append(concatenated_features)
    return np.concatenate(all_features)

def adjust_weights_based_on_reliability(features, reliability_scores):
    """ Adjust weights based on landmark reliability. """
    # Simple weighted sum of features based on reliability scores (PLACEHOLDER)
    weighted_features = [feat * score for feat, score in zip(features, reliability_scores)]
    return sum(weighted_features)

def process_face(gray_img, landmarks):
    features = []

    for landmark_name, position in landmarks.items():
        region = get_region(gray_img, position, size=40)

        if region is None or region.size == 0:
            continue
        
        features = extract_features(region)

    # Aggregate features from all regions
    if features:
        aggregated_features = np.concatenate(features)
    else:
        aggregated_features = np.array([])

    return aggregated_features

# Load image and convert to grayscale
image_path = '../test/assets/aflw-test.jpg'
input_image = io.imread(image_path)
gray_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

""" Region-based around 5 landmarks of detected face. """
# Example data (from retinaFace structure)
faces_data = {
    "face_1": {
        "landmarks": {
            "right_eye": [257.8, 209.6],
            "left_eye": [374.9, 251.8],
            "nose": [303.5, 299.9],
            "mouth_right": [228.4, 338.7],
            "mouth_left": [320.2, 374.6]
        }
    }
}

for face_id, face_info in faces_data.items():
    local_feature_descriptor = process_face(gray_img, face_info['landmarks'])
    print(f"Features for {face_id}: {local_feature_descriptor}")