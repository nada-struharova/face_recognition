import numpy as np
import os
import cv2
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.patches import Rectangle

# ------------------ Dataset Loading ------------------ 
def load_dataset(dataset_path):
    """Loads images from your dataset structure."""
    images = []
    identities = []

    for identity_dir in os.listdir(dataset_path):
        identity_path = os.path.join(dataset_path, identity_dir)

        # Check if it's actually a directory
        if not os.path.isdir(identity_path):
            continue  # Skip non-directory entries

        for image_filename in os.listdir(identity_path):
            image_path = os.path.join(identity_path, image_filename)

            # Only process files, not directories or hidden files
            if os.path.isfile(image_path) and not image_filename.startswith('.'):
                image = cv2.imread(image_path)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                identities.append(identity_dir)

    return images, identities

# ------------------ Image Preprocessing ------------------ 
def preprocess_face(image, face_data):
    """ Preprocesses a face image using RetinaFace data.

    Args:
        image: The original image.
        face_data: RetinaFace output for a face  
                   (containing 'landmarks' and 'facial_area').

    Returns:
        The preprocessed face image. 
    """
    facial_area = face_data['facial_area']
    landmarks = face_data['landmarks']

    left, top, right, bottom = facial_area 
    width = right - left
    height = bottom - top

    # Margin to account for potential rotation errors
    margin_factor = 0.1
    margin_x = int(width * margin_factor)
    margin_y = int(height * margin_factor)

    # Crop the image
    x_offset = max(0, left - margin_x)  # Actual shift in x-coordinate
    y_offset = max(0, top - margin_y)  # Actual shift in y-coordinate
    # face_img = image[max(0, top - margin_y):min(image.shape[0], bottom + margin_y), 
    #                 max(0, left - margin_x):min(image.shape[1], right + margin_x)]
    face_img = image[y_offset:min(image.shape[0], bottom + margin_y), 
                    x_offset:min(image.shape[1], right + margin_x)]

    # Update landmarks after cropping
    new_landmarks = {}
    for landmark_name, (x, y) in landmarks.items():
        # new_landmarks[landmark_name] = (x - (left - margin_x), y - (top - margin_y))
        new_landmarks[landmark_name] = (x - x_offset, y - y_offset)

    # Alignment based on eyes
    left_eye_x, left_eye_y = landmarks['left_eye']
    right_eye_x, right_eye_y = landmarks['right_eye']

    # Horizontal alignemnt
    dy = right_eye_y - left_eye_y
    dx = right_eye_x - left_eye_x
    angle = np.arctan2(dy, dx)
    center = (((left_eye_x + right_eye_x) // 2), ((left_eye_y + right_eye_y) // 2))

    # Rotate cropped image with padding
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_img = cv2.warpAffine(face_img, rotation_matrix, face_img.shape[1::-1], 
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Update landmarks after rotation
    for landmark_name, (x, y) in new_landmarks.items():
        # Translate before rotating
        x -= center[0]
        y -= center[1]

        # Rotate
        original_pos = np.array([x, y, 1]) # homogenougs coordinates
        new_pos = rotation_matrix.dot(original_pos)

        # Translate back
        new_pos[0] += center[0]
        new_pos[1] += center[1]

        new_landmarks[landmark_name] = (int(new_pos[0]), int(new_pos[1]))

    # Normalise and Resize
    output_size = (224, 224)
    normalised_img = normalise_image(aligned_img, output_size)
    
    # Update landmarks after normalization and resize
    scale_x = output_size[0] / aligned_img.shape[1]
    scale_y = output_size[1] / aligned_img.shape[0]
    new_landmarks = {key: (int(value[0] * scale_x), int(value[1] * scale_y))
                         for key, value in new_landmarks.items()}

    return normalised_img, new_landmarks

def normalise_image(image, output_size):
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the desired output size
    resized_img = cv2.resize(gray_img, output_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize the pixel values to the range [0, 1]
    # normalised_img = resized_img.astype(np.float32) / 255.0
    
    return resized_img

# ------------------ Split Image into Grid / Landmark-Centric Regions ------------------ 
def split_into_grid(image, grid_size=(6, 6)):
   height, width = image.shape[:2]
   grid_rows, grid_cols = grid_size
   row_size = height // grid_rows
   col_size = width // grid_cols 

   regions = []
   for i in range(grid_rows):
       for j in range(grid_cols):
            # Split logic
            x1, y1 = j * col_size, i * row_size
            x2, y2 = x1 + col_size, y1 + row_size  
            region = image[y1:y2, x1:x2]

            # Error Handling: Type mismatch
            if region.dtype != np.uint8:
                region = region.astype(np.uint8)

            regions.append(region)

   return regions

def split_into_regions_landmarks_fixed_size(image, landmarks, region_sizes):
    """Splits an image into fixed-size regions based on facial landmarks.

    Args:
        image: A NumPy array representing the image.
        landmarks: A NumPy array of shape (5, 2) containing the coordinates of 5 facial landmarks (x, y).
        region_sizes: A dictionary specifying the width and height for each region type. 
                      Example: {'eye': (50, 30), 'nose': (40, 40), 'mouth': (60, 40)}

    Returns:
        A list of NumPy arrays, each representing a fixed-size region of the face.
    """

    # Define region centers based on landmarks
    left_eye_center = landmarks[0]
    right_eye_center = landmarks[1]
    nose_tip = landmarks[2]
    mouth_left = landmarks[3]
    mouth_right = landmarks[4]

    regions = []

    for landmark_name, landmark_coord in zip(['left_eye', 'right_eye', 'nose', 'mouth'], 
                                             [left_eye_center, right_eye_center, nose_tip, mouth_left]):
        # Get the region size from the dictionary
        region_width, region_height = region_sizes[landmark_name]

        # Calculate the starting coordinates, ensuring the region stays within the image bounds
        region_x = max(0, int(landmark_coord[0] - region_width / 2))
        region_y = max(0, int(landmark_coord[1] - region_height / 2))

        # Extract the region, handling cases where it goes beyond image boundaries
        region = image[region_y : min(region_y + region_height, image.shape[0]),
                       region_x : min(region_x + region_width, image.shape[1])]

        regions.append(region)

    return regions

def split_into_landmark_regions(image, landmarks, expansion_factor=1.2):
    """Splits an image into regions based on facial landmarks.

    Args:
        image: A NumPy array representing the image (224x224).
        landmarks: A NumPy array of shape (5, 2) containing the coordinates of 5 facial landmarks (x, y).
        expansion_factor: A factor to expand the region boundaries for flexibility.

    Returns:
        A list of NumPy arrays, each representing a region of the face.
    """

    # Define region centers based on landmarks
    left_eye = np.array(landmarks['left_eye'])
    right_eye = np.array(landmarks['right_eye'])
    nose = np.array(landmarks['nose'])
    mouth_left = np.array(landmarks['mouth_left'])
    mouth_right = np.array(landmarks['mouth_right'])

    # Calculate region boundaries with expansion
    eye_height = max(np.linalg.norm(left_eye - nose),
                            np.linalg.norm(right_eye - nose)) * expansion_factor
    eye_width = np.linalg.norm(left_eye - right_eye) * expansion_factor
    mouth_height = np.linalg.norm(nose - mouth_left) * expansion_factor * 2  # doubles for mouth size
    mouth_width = np.linalg.norm(mouth_left - mouth_right) * expansion_factor

    regions = []

    # Left Eye Region
    left_eye_x = int(left_eye[0] - eye_height / 2)
    left_eye_y = int(left_eye[1] - eye_height / 2)
    regions.append(image[left_eye_y : left_eye_y + int(eye_height), left_eye_x : left_eye_x + int(eye_width)])

    # Right Eye Region
    right_eye_x = int(right_eye[0] - eye_width / 2)
    right_eye_y = int(right_eye[1] - eye_height / 2)
    regions.append(image[right_eye_y : right_eye_y + int(eye_height), right_eye_x : right_eye_x + int(eye_width)])

    # Nose Region
    nose_x = int(nose[0] - eye_width / 2)
    nose_y = int(nose[1])
    regions.append(image[nose_y : nose_y + int(eye_height), nose_x : nose_x + int(eye_width)])

    # Mouth Region
    mouth_x = int(mouth_left[0] - mouth_width / 2)
    mouth_y = int(mouth_left[1])
    regions.append(image[mouth_y : mouth_y + int(mouth_height), mouth_x : mouth_x + int(mouth_width)])

    return regions

def extract_roi_around_landmarks(image, landmarks, roi_size=56, padding=10):
    """
    Extracts ROIs around landmarks, handling edge cases and errors.

    Args:
        image: The input grayscale image (224x224, normalized).
        landmarks: A numpy array of shape (5, 2) containing landmark coordinates (x, y).
        roi_size: The desired size of each ROI (square).
        padding: Additional padding around each landmark.

    Returns:
        rois: A list of image patches (ROIs) as numpy arrays, or None if errors occur.
    """

    rois = []
    for landmark in landmarks:
        x, y = landmark

        # Error Handling
        # Check if landmark is within image bounds
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            print(f"Error: Landmark ({x}, {y}) is outside image bounds.")
            return None  # Return None to signal error

        # Get ROI boundaries with padding
        x1 = max(0, int(x - roi_size / 2 - padding))
        y1 = max(0, int(y - roi_size / 2 - padding))
        x2 = min(image.shape[1], int(x + roi_size / 2 + padding))
        y2 = min(image.shape[0], int(y + roi_size / 2 + padding))

        # Extract ROI
        roi = image[y1:y2, x1:x2]

        # Error Handling
        # Check if ROI is empty
        if roi.size == 0:
            print(f"Error: Empty ROI for landmark ({x}, {y}).")
            return None  # Return None to signal error

        # Resize ROI and interpolate
        if roi.shape[0] != roi_size or roi.shape[1] != roi_size:
            roi = cv2.resize(roi, (roi_size, roi_size), interpolation=cv2.INTER_AREA)  # Use INTER_AREA for shrinking

        rois.append(roi)

    return rois

def visualise_rois(image, landmarks, rois):
    """
    Visualizes the image with landmarks and extracted ROIs.

    Args:
        image: The input image.
        landmarks: A numpy array of landmark coordinates.
        rois: A list of ROIs.
    """
    plt.imshow(image, cmap='gray')
    for landmark in landmarks:
        plt.scatter(landmark[0], landmark[1], color='red', marker='x')

    for roi, landmark in zip(rois, landmarks):
        x, y = landmark
        x1 = int(x - roi.shape[1] / 2)
        y1 = int(y - roi.shape[0] / 2)
        plt.gca().add_patch(Rectangle((x1, y1), roi.shape[1], roi.shape[0], linewidth=2, edgecolor='green', facecolor='none'))
    plt.show()

# ------------------ Storing Feature Vectors and Descriptors ------------------ 
def save_features(features_dict, filename="extracted_features.pkl"):
    """Saves feature vectors to a file using pickle.

    Args:
        features_dict (dict): Dictionary containing the extracted feature vectors.
        filename (str): Name of the file to save (.pkl extension recommended). Defaults to "extracted_features.pkl".
    """
    with open(filename, 'wb') as f:
        pickle.dump(features_dict, f)

def load_features(filename="extracted_features.pkl"):
    """Loads feature vectors from a file saved using pickle.

    Args:
        filename (str): Name of the file to load. Defaults to "extracted_features.pkl".

    Returns:
        dict: The loaded dictionary of feature vectors.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

# ------------------ Validation & Analysis ------------------ 
def calculate_l2_dist_euclidian(features_dict): 
    """Calculates pairwise L2 distances between feature descriptors.
    """
    distances = {'intra_distances': [], 'inter_distances': []}

    for identity1, features_list1 in features_dict.items():
        for identity2, features_list2 in features_dict.items():
            if identity1 == identity2:  # Intra-Identity Comparison
                for i in range(len(features_list1)):
                    for j in range(i + 1, len(features_list1)): 
                        dist = np.linalg.norm(features_list1[i] - features_list2[j])  
                        distances['intra_distances'].append(dist)  

            else:  # Inter-Identity Comparison
                for features1 in features_list1:
                    for features2 in features_list2:
                        dist = np.linalg.norm(features1 - features2)
                        distances['inter_distances'].append(dist)

    return distances

def calculate_l2_dist_with_baselines(features_dict): 
    raw_dist = {'intra_distances': [], 'inter_distances': []}

    random_dist = {'intra_distances': [], 'inter_distances': []}

    for identity1, features_list1 in features_dict.items():
        for identity2, features_list2 in features_dict.items():
            if identity1 == identity2: 
                for i in range(len(features_list1)):
                    for j in range(i + 1, len(features_list1)): 
                        # Raw Features
                        dist_raw = np.linalg.norm(features_list1[i][0] - features_list2[j][0])  
                        raw_dist['intra_distances'].append(dist_raw)  

                        # Random Features
                        dist_random = np.linalg.norm(features_list1[i][1] - features_list2[j][1])
                        random_dist['intra_distances'].append(dist_random)

            else:  
                for features1 in features_list1:
                    for features2 in features_list2:
                        # Raw Features 
                        dist_raw = np.linalg.norm(features1[0] - features2[0])
                        raw_dist['inter_distances'].append(dist_raw)  

                        # Random Features
                        dist_random = np.linalg.norm(features1[1] - features2[1])
                        random_dist['inter_distances'].append(dist_random)

    return raw_dist, random_dist

def analyse_features(distances):
    # ... (Add more sophisticated analysis here: means, distributions, visualization)
    print("Intra-Identity Distances:")
    print("  Mean:", np.mean(distances['intra_distances']))
    print("  Standard Deviation:", np.std(distances['intra_distances']))
    print("  Median:", np.median(distances['intra_distances']))
    print("  Min:", np.min(distances['intra_distances']))
    print("  Max:", np.max(distances['intra_distances']))

    print("\nInter-Identity Distances:")
    print("  Mean:", np.mean(distances['inter_distances']))
    print("  Standard Deviation:", np.std(distances['inter_distances']))
    print("  Median:", np.median(distances['inter_distances']))
    print("  Min:", np.min(distances['inter_distances']))
    print("  Max:", np.max(distances['inter_distances']))

# ------------------ Visualisation ------------------ 
def visualize_preprocessed_face(image, landmarks):
    """
    1. Display original image with landmarks detected via RetinaFace,
    2. Display preprocessed face image with transformed landmarks.

    Args:
        image: The original image (for reference).
        face_data: RetinaFace output.
        preprocessed_image: Preprocessed face image.
        resized_landmarks: Dictionary containing the resized landmark positions.
    """
    image_with_landmarks = image.copy()
    new_radius = 3  # Consider slightly bigger radius on the smaller preprocessed img
    new_color = (0, 255, 0)
    for _, (x, y) in landmarks.items():
        x = int(x)  # Cast x to integer
        y = int(y)  # Cast y to integer
        cv2.circle(image_with_landmarks, (x, y), new_radius, new_color, -1)
    cv2.imshow("Preprocessed Face with Landmarks", image_with_landmarks)

    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()

def histograms(distances):
    """ Show distribution of distances within the same identity and between different identities.
        - intra-identity distances should be tightly clustered with low values 
        - inter-identity distances should be larger and more spread out. """ 
    plt.figure(figsize=(10, 5))

    plt.subplot(1,2,1)
    plt.hist(distances['intra_distances'])
    plt.xlabel('L2 Distance')
    plt.ylabel('Frequency')
    plt.title('Intra-Identity Distances')

    plt.subplot(1,2,2)
    plt.hist(distances['inter_distances'])
    plt.xlabel('L2 Distance')
    plt.ylabel('Frequency')
    plt.title('Inter-Identity Distances')

    plt.show() 

    # plt.hist(distances['intra_distances'], bins=20, label='Intra-Identity')
    # plt.hist(distances['inter_distances'], bins=20, label='Inter-Identity')
    # plt.legend()
    # plt.show()

def discriminative_power(true_labels, decision_scores):
    """
    Args:
        true_labels: Pairs of images and labels (1 for intra-identity, 0 for otherwise).
        decision_scores: Calculated distances.
    """
    fpr, tpr, thresholds = roc_curve(true_labels, decision_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--') # Random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

image = cv2.imread('face_recognition/datasets/evaluate_local/Naty/001.jpeg')

