import face_alignment
from skimage import io, gabor, img_as_float
import cv2
import numpy as np
import utils

def define_rois(image, landmarks, feature_extractor):
    """ Define ROIs from 68-point landmark model. """
    regions = {
        'left_eye': landmarks[36:42],  # Left eye
        'right_eye': landmarks[42:48],  # Right eye
        'nose': landmarks[27:36],      # Nose bridge and tip
        'mouth': landmarks[48:68]      # Mouth
    }
    features = {}
    
    for key, points in regions.items():
        # Create a mask to extract region
        mask = np.zeros_like(image)
        points = np.array(points, dtype=np.int32)
        cv2.fillConvexPoly(mask, points, (255, 255, 255))
        
        # Extract region (masking)
        region = cv2.bitwise_and(image, mask)
        x, y, w, h = cv2.boundingRect(points)
        roi = region[y:y+h, x:x+w]
        
        # Detect and compute features in the ROI
        keypoints, descriptors = feature_extractor.detectAndCompute(roi, None)
        features[key] = (keypoints, descriptors)
    
    return features

# SIFT features
def extract_sift_features(image, keypoints, radius=15):
    sift = cv2.SIFT_create()
    regions = [cv2.KeyPoint(x=keypoint[0], y=keypoint[1], _size=radius*2) for keypoint in keypoints]
    _, descriptors = sift.compute(image, regions)
    return descriptors

# SURF features (faster for real-time applications without sacrificing too much)
def extract_surf_features(image, keypoints, radius=15, hessianThreshold=400):
    """ Extract SURF features from 68 landmarks. """
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold)
    keypoint_regions = [cv2.KeyPoint(x=keypoint[0], y=keypoint[1], _size=radius*2) for keypoint in keypoints]
    _, descriptors = surf.compute(image, keypoint_regions)
    return descriptors

def extract_hog_features(image, keypoints, radius=15, cells_per_block=(2, 2), pixels_per_cell=(16, 16)):
    """ Extract HOG features from 68 landmarks. """
    hog_descriptors = []
    for keypoint in keypoints:
        x, y = int(keypoint[0]), int(keypoint[1])
        patch = image[y-radius:y+radius, x-radius:x+radius]
        if patch.shape[0] == 2*radius and patch.shape[1] == 2*radius:
            descriptor = cv2.HOGDescriptor(_winSize=(patch.shape[1], patch.shape[0]),
                                           _blockSize=(2*pixels_per_cell[0], 2*pixels_per_cell[1]),
                                           _blockStride=(pixels_per_cell[0], pixels_per_cell[1]),
                                           _cellSize=(pixels_per_cell[0], pixels_per_cell[1]),
                                           _nbins=9)
            hog_descriptors.append(descriptor.compute(patch))
    return np.array(hog_descriptors).squeeze()

def extract_gabor_features(image, keypoints, frequency=0.6, radius=15):
    """ Extract Gabor features from 68 landmarks. """
    gabor_descriptors = []
    image = img_as_float(image)
    for keypoint in keypoints:
        x, y = int(keypoint[0]), int(keypoint[1])
        patch = image[y-radius:y+radius, x-radius:x+radius]
        if patch.shape[0] == 2*radius and patch.shape[1] == 2*radius:
            filt_real, filt_imag = gabor(patch, frequency=frequency)
            gabor_descriptors.append(filt_real.flatten())
    return np.array(gabor_descriptors)

# Load image and convert to grayscale
input_image = io.imread('../test/assets/aflw-test.jpg')
gray_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

# Initialize 68 landmark detection and face alignment
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

# Get 2D landmarks
landmarks = fa.get_landmarks_from_image(input_image)

# Extract local feature descriptors
surf_features = extract_surf_features(input_image, landmarks)
hog_features = extract_hog_features(input_image, landmarks)

# Normalise local features
surf_normalised = utils.normalise_features(surf_features)
hog_normalised = utils.normalise_features(hog_features)

# Concatenate all local descriptors to form a feature vector
local_descriptor = utils.concatenate_features(surf_normalised, hog_normalised)

print("SURF features:", surf_features.shape)
print("HOG features:", hog_features.shape)

# # ROI approach
# sift = cv2.SIFT_create()
# if landmarks is not None:
#     first_face_landmarks = landmarks[0]  # Using the first face's landmarks
#     rois = define_rois(gray_image, first_face_landmarks, sift)

#     # Example: Print the number of SIFT features found in each region
#     for region, (kp, desc) in rois.items():
#         if desc is not None:
#             print(f"Number of features in {region}: {len(desc)}")
#         else:
#             print(f"No features detected in {region}")

cv2.imshow("Processed Image", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
