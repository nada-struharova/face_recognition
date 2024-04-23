from skimage import io, gabor, img_as_float
import cv2
import numpy as np
import utils

def get_region(image, center, size=20): # TODO: size 20 is arbitrary, can adjust
    """ Extract a square region around a given facial landmark. """

    x, y = int(center[0]), int(center[1])
    x1, y1, x2, y2 = x - size, y - size, x + size, y + size
    if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        return None  # Region is out of image bounds
    return image[y1:y2, x1:x2]

def sift_features(image_region):
    """Extract SIFT features from an image region.
    Args:
        image_region: A grayscale image region

    Returns:
        Numpy array containing flattened SIFT features. 
    """

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_region, None)

    if descriptors is None:  # Handle exceptions
        descriptors = np.zeros((1, 128), dtype=np.float32)  # Default SIFT descriptor size

    return descriptors.flatten()  # Flatten for concatenation 

def surf_features(image, hessianThreshold=400):
    """ Extract SURF features from an face region.
    Args:
        image_region: A grayscale image region

    Returns:
        Numpy array containing flattened SURF features.
    """

    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
    _, descriptors = surf.detectAndCompute(image, None)

    if descriptors is None: # Handle exceptions
        descriptors = np.zeros((1, surf.descriptorSize()), dtype=np.float32)

    return descriptors.flatten() # Flatten for concatenation 

def hog_features(image, win=(64, 64), block=(16,16), stride=(8,8), cell=(8,8), bins=9):
    """ Extract HOG features from an face region.
    Args:
        image_region: A grayscale image region

    Returns:
        Numpy array containing flattened SURF features.
    """
    hog = cv2.HOGDescriptor(win, block, stride, cell, bins)
    hog_features = hog.compute(image)
    return hog_features.flatten() # Flatten for concatenation 

def region_local_features(region, sift=True, surf=True, hog=True, normalise=True):
    """ Extract local features of region and concatenate. """
    features = []

    if sift:
        sift_features = sift_features(region) 
        features.append(sift_features)
    if surf:
        surf_features = surf_features(region)
        features.append(surf_features)
    if hog:
        hog_features = hog_features(region)
        features.append(hog_features)

    if normalise:
        for i in range(len(features)):
            features[i] /= np.linalg.norm(features[i], ord=2)  

    return np.concatenate(features)

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

def extract_local_features(gray_img, landmarks, sift=True, surf=True, hog=True, normalise=True):
    """ Extract all wanted local features. """
    features = []

    for landmark_name, position in landmarks.items():
        region = get_region(gray_img, position)

        if region is None or region.size == 0:
            continue
        
        # TODO: change args to experiment with different local features extracted
        region_features = region_local_features(region, sift, surf, hog, normalise)
        features.append(region_features)

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
    local_feature_descriptor = extract_local_features(gray_img, face_info['landmarks'])
    print(f"Features for {face_id}: {local_feature_descriptor}")