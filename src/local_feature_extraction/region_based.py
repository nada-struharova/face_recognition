import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2.xfeatures2d
from retinaface import RetinaFace
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve, auc
from skimage.feature import local_binary_pattern

# ------------------ Utilities ------------------ 
def split_into_grid(image, grid_size=(3, 3)):
   height, width = image.shape[:2]
   grid_rows, grid_cols = grid_size
   row_size = height // grid_rows
   col_size = width // grid_cols 

   regions = []
   for i in range(grid_rows):
       for j in range(grid_cols):
           x1, y1 = j * col_size, i * row_size
           x2, y2 = x1 + col_size, y1 + row_size  
           regions.append(image[y1:y2, x1:x2])

   return regions

def split_into_landmark_regions(image, bounding_box, landmarks, region_count=5):
    """ Divide the image into regions based on facial landmarks.
    Args:
        image: Image to extract regions from.
        bounding_box: Bounding box coordinates as (x1, y1, x2, y2).
        landmarks: Dictionary of landmark coordinates.
        region_count: The number of regions (default 5, corresponding to the number of key landmarks).
    Returns:
        List of Image Regions, each centered around a landmark. """

    x1, y1, x2, y2 = bounding_box
    regions = []
    
    # Determine the dimensions of each region based on the bounding box
    width = x2 - x1
    height = y2 - y1
    region_width = width // 2
    region_height = height // 2

    # Define centers of regions around landmarks
    landmark_keys = list(landmarks.keys())
    for key in landmark_keys:
        cx, cy = landmarks[key]
        # Calculate region coordinates ensuring they are within the image boundaries
        reg_x1 = max(x1, int(cx - region_width / 2))
        reg_y1 = max(y1, int(cy - region_height / 2))
        reg_x2 = min(x2, reg_x1 + region_width)
        reg_y2 = min(y2, reg_y1 + region_height)
        
        # Ensure the region is within the bounds of the image
        reg_x1 = max(0, min(reg_x1, image.shape[1] - region_width))
        reg_y1 = max(0, min(reg_y1, image.shape[0] - region_height))
        reg_x2 = reg_x1 + region_width
        reg_y2 = reg_y1 + region_height

        region = image[reg_y1:reg_y2, reg_x1:reg_x2]
        if region.size == 0:
            print(f"Warning: Region is empty for landmark: {key}")
            continue
        regions.append((region, key))

    return regions

def split_landmark_grid(image, bbox, landmarks, grid_size=(5, 5)):
    """ Extract regions from the facial area based on a landmark-guided grid. """
    def find_landmarks_in_region(landmarks, x1, y1, x2, y2):
        region_landmarks = []
        for landmark_name, (x, y) in landmarks.items():
            if x1 <= x <= x2 and y1 <= y <= y2:  
                region_landmarks.append((landmark_name, (x, y)))  

        return region_landmarks

    top, left, bottom, right = bbox
    width = right - left
    height = bottom - top

    grid_rows, grid_cols = grid_size
    row_size = height // grid_rows
    col_size = width // grid_cols

    regions = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            x1, y1 = left + j * col_size, top + i * row_size
            x2, y2 = x1 + col_size, y1 + row_size

            # Adjust region boundaries based on nearby landmarks (if needed)
            region_landmarks = find_landmarks_in_region(landmarks, x1, y1, x2, y2)
            if region_landmarks:  
                # Modify x1, y1, x2, y2 based on landmark positions (add overlap if desired)
                pass  # Implement your adjustment logic here

            regions.append(image[y1:y2, x1:x2])

    return regions

def normalise_image(image, output_size):
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the desired output size
    resized_img = cv2.resize(gray_img, output_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize the pixel values to the range [0, 1]
    normalised_img = resized_img.astype(np.float32) / 255.0
    
    return normalised_img

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

# ------------------ Local Feature Extraction ------------------ 
def extract_sift_features(image_region):
    """Extract SIFT features from an image region.
    Args:
        image_region: A grayscale image region

    Returns:
        Numpy array containing flattened SIFT features. 
    """

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_region, None)

    print("Number of SIFT keypoints detected:", len(keypoints))

    if descriptors is None:  # Handle exceptions
        descriptors = np.zeros((1, 128), dtype=np.float32)  # Default SIFT descriptor size

    return descriptors.flatten()  # Flatten for concatenation 

def extract_surf_features(image, hessianThreshold=400):
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

def extract_hog_features(image, win=(32, 32), block=(8,8), stride=(4,4), cell=(4,4), bins=9):
    """ Extract HOG features from an face region.
    Args:
        image_region: A grayscale image region

    Returns:
        Numpy array containing flattened SURF features.
    """
    hog = cv2.HOGDescriptor(win, block, stride, cell, bins)
    hog_features = hog.compute(image)
    return hog_features.flatten().ravel() # Flatten for concatenation 

def extract_lbp_features(image, radius=3, num_points=8):
    """Extract LBP features from a face region.
    Args:
        image_region: A grayscale image region
        radius: Radius for the LBP circular neighborhood
        num_points: Number of points in LBP circular neighborhood
    Returns:
        Numpy array containing LBP histogram (feature vector)
    """
    num_points = num_points * radius
    lbp = local_binary_pattern(image, num_points, radius, method='uniform') 
    n_bins = int(lbp.max() + 1)  # Uniform LBP has a fixed range of values
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    print("Dimension of LBP descriptor:", hist.shape)
    return hist

def average_landmark_distance(landmarks):
    landmark_points = list(landmarks.values())
    num_points = len(landmark_points)
    total_distance = 0
    count = 0

    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(np.array(landmark_points[i]) - np.array(landmark_points[j]))
            total_distance += dist
            count += 1

    return total_distance / count if count != 0 else 0

def calculate_region_size(facial_area, landmarks):
    x1, y1, x2, y2 = facial_area
    box_width = x2 - x1
    box_height = y2 - y1
    average_dimension = (box_width + box_height) / 2

    # Get average distance between landmarks
    average_distance = average_landmark_distance(landmarks)

    # Calculate region size based on a smaller ratio of average_distance or a fraction of average_dimension
    return int(min(average_distance, average_dimension * 0.2))

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
    face_img = image[y_offset:min(image.shape[0], bottom + margin_y), 
                    x_offset:min(image.shape[1], right + margin_x)]

    # Update landmarks after cropping
    new_landmarks = {}
    for landmark_name, (x, y) in landmarks.items():
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

    # face_img = image[max(0, top - margin_y):min(image.shape[0], bottom + margin_y), 
    #                 max(0, left - margin_x):min(image.shape[1], right + margin_x)]

    # # Update landmarks after cropping
    # new_landmarks = {}
    # for landmark_name, (x, y) in landmarks.items():
    #     new_landmarks[landmark_name] = (x - (left - margin_x), y - (top - margin_y)
    
    # Update landmarks after rotation    
    # new_landmarks = {}
    # for landmark_name, (x, y) in landmarks.items():
    #     original_pos = np.array([x, y, 1])  # Add homogeneous coordinate
    #     new_pos = rotation_matrix.dot(original_pos)
    #     new_landmarks[landmark_name] = new_pos[:2].astype(int)

def set_region(image, center, size=20):
    """ Extract a square region around a given facial landmark.
    Args:
        image: Image to extract region from. 
        center: Centre coordinates of the wanted region.
        size: Size of the region. (arbitrary default value = 20) 
    Returns:
        Image Region. """

    x, y = int(center[0]), int(center[1])
    x1, y1, x2, y2 = x - size, y - size, x + size, y + size

    print(f"Attempting to extract region at ({x1}, {y1}) to ({x2}, {y2})")  # Debug dimensions

    if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        print(f"Warning: Region out of bounds for center: {center}, size: {size}") 
        print(f"Image shape: {image.shape}")
        return None 
    
    region = image[y1:y2, x1:x2]
    print(f"Extracted Region Size: {region.shape}")  # Show actual extracted size
    return region

def get_region_features(region, sift=True, surf=False, hog=False, lbp=True):
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
        sift_feat = extract_sift_features(region)
        if not isinstance(sift_feat, np.ndarray):
            sift_feat = np.array(sift_feat)  # Convert to np array
        features = np.concatenate([features, sift_feat])  # Concatenate NumPy arrays
    if surf:
        surf_feat= extract_surf_features(region)
        if not isinstance(surf_feat, np.ndarray):
            surf_feat = np.array(surf_feat)  # Convert to np array
        features = np.concatenate([features, surf_feat])
    if hog:
        hog_feat = extract_hog_features(region)
        if not isinstance(hog_feat, np.ndarray):
            hog_feat = np.array(hog_feat)  # Convert to np array
        features = np.concatenate([features, hog_feat])
    if lbp:
        lbp_feat = extract_lbp_features(region)
        if not isinstance(lbp_feat, np.ndarray):
            lbp_feat = np.array(lbp_feat)  # Convert to np array
        features = np.concatenate([features, lbp_feat])

    print(f"Region shape of features: {features.shape}")
    return features

def extract_features(image, faces_data):
    features = {}

    for face_id, data in faces_data.items():
        facial_area = data['facial_area']  # Assuming 'facial_area' has been added to each face's data
        landmarks = data['landmarks']

        # Calculate dynamic region size based on landmarks and facial area
        region_size = calculate_region_size(facial_area, landmarks)

        # Now use this region_size for further processing
        for landmark_name, position in landmarks.items():
            region = set_region(image, position, size=region_size)
            if region is not None:
                # Extract features from region
                region_features = get_region_features(region)
                features[landmark_name] = region_features

    return features

def extract_from(image):
    """ Extracts local features using RetinaFace. 
    
    Args:
        image: Image to detect faces and lanmdarks in. 
        
    Returns:
        Face data (landmarks, bounding box, features, etc.) on all detected faces. """
    
    # Detect all faces, face data (landmarks, facial area) in image
    faces = RetinaFace.detect_faces(image)  

    # Each face: preprocess -> extract local features -> store
    faces_features = {}  
    for face_id, face_data in faces.items():
        face_img, landmarks = preprocess_face(image, face_data)      

        # Extract and store local features
        region_features = extract_local_feats(face_img, face_data['facial_area'], landmarks)
        faces_features.setdefault(face_id, []).append(region_features)  

    # Return all faces with per-face extracted features
    return faces_features

# ------------------ Extracton ------------------ 
def extract_local_features(gray_img, landmarks, sift=True, surf=False, hog=True, lbp=True, region_norm=False):
    """ Extract all wanted local features. """
    # Adaptive region size -> Sets region size based on avg landmark distance
    left_eye_x, left_eye_y = landmarks['left_eye']
    right_eye_x, right_eye_y = landmarks['right_eye']
    distance = np.sqrt(((right_eye_x - left_eye_x) ** 2) + ((right_eye_y - left_eye_y) ** 2)) 
    region_size = int(0.6 * distance)

    features = np.array([])
    for landmark_name, position in landmarks.items():
        # Set region
        region = set_region(gray_img, position, size=region_size)

        # Error handling
        if region is None or region.size == 0:
            print("Error: Empty region for landmark:", landmark_name, position)
            continue

        print("Landmark:", landmark_name, "Region Size:", region.size) # Debugging

        # Extract Local Features
        region_features = get_region_features(region, sift, surf, hog, lbp)
        print("Region features: ", region_features)

        # Individual normalisation
        if region_norm:
            for i in range(len(region_features)):
                region_features[i] /= np.linalg.norm(region_features[i], ord=2) 
        
        # Concatenate directly
        if features.size == 0:
            features = region_features
        else:
            features = np.concatenate([features, region_features]) 

    # Global Normalization
    if not region_norm:
        features /= np.linalg.norm(features, ord=2) 

    return features

def extract_local_feats(gray_img, facial_area, landmarks, sift=True, surf=False, hog=False, lbp=True, region_norm=False):
    """ Extract all wanted local features. """

    regions = split_into_landmark_regions(gray_img, facial_area, landmarks)
    features = np.array([])

    for region, landmark in regions:
        # # Error handling
        if region is None or region.size == 0:
            print("Error: Empty region for landmark!")
            continue

        # Extract Local Features
        region_features = get_region_features(region, sift, surf, hog, lbp)
        print("Region features: ", region_features)

        # Individual normalisation
        if region_norm:
            for i in range(len(region_features)):
                region_features[i] /= np.linalg.norm(region_features[i], ord=2) 
        
        # Concatenate directly
        if features.size == 0:
            features = region_features
        else:
            features = np.concatenate([features, region_features]) 

    # Global Normalization
    if not region_norm:
        features /= np.linalg.norm(features, ord=2) 

    print("Total descriptor size after concatenation:", features.shape)
    return features

def extract_features_from_regions(regions, sift=True, surf=False, hog=True, lbp=True):
    all_features = []
    for region in regions:
        # Your code to extract SIFT, HOG, LBP features from a single region
        region_features = get_region_features(region, sift, surf, hog, lbp)
        all_features.append(region_features)
    return all_features

# ------------------ Dynamic Weighting Scheme ------------------ 
def adjust_weights_based_on_reliability(features, reliability_scores):
    """ Adjust weights based on landmark reliability. """
    # Simple weighted sum of features based on reliability scores (PLACEHOLDER)
    weighted_features = [feat * score for feat, score in zip(features, reliability_scores)]
    return sum(weighted_features)

# ------------------ Validation ------------------ 
def calculate_l2_dist(features_dict): 
    """Calculates pairwise L2 distances between feature descriptors.
    """
    distances = {'intra_distances': [], 'inter_distances': []}

    for identity1, features1 in features_dict.items():
        for identity2, features2 in features_dict.items():
            if identity1 == identity2:  
                dists = euclidean_distances(features1, features1) 
                # Extract distances below diagonal (excluding self-comparisons)
                for i in range(1, len(features1)):  
                    for j in range(i): 
                        distances['intra_distances'].append(dists[i][j]) 

            else:  
                dists = euclidean_distances(features1, features2)
                for dist in dists.flatten(): # Flatten to add all distances
                    distances['inter_distances'].append(dist)

    return distances

def analyze_features(distances):
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

# ------------------ Dataset Loading and Processing ------------------ 
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
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                identities.append(identity_dir)

    return images, identities

# ------------------ Testing logic ------------------ 
# Dataset loading and feature extraction
# images, identities = load_dataset("face_recognition/datasets/evaluate_local") 
# features = {} 

# for image, identity in zip(images, identities):
#     if image is None:
#         print("Image: ", image)
#     print("ID: ", identity)
#     feat_dict = extract_from(image)
#     features.update(feat_dict) 

# # Feature intra- and inter- distance
# final_distances = calculate_l2_dist(features)

# # ------------------ Analysis & Visualisation ------------------ 

# # Metrics and Distribution Analysis
# analyze_features(final_distances)

# # Histograms of inter- and intra- distances
# histograms(final_distances)

image = cv2.imread('face_recognition/datasets/evaluate_local/Nada/025.jpeg')  
faces = RetinaFace.detect_faces(image)  

for face_id, face_data in faces.items():
    face_img, resized_landmarks = preprocess_face(image.copy(), face_data)

    print("Original Image Shape: ", image.shape)
    print("Original landmarks: ", face_data['landmarks'])
    print("Resized Image Shape: ", face_img.shape)
    for landmark_name, (x, y) in resized_landmarks.items():
        print(f"New Landmark: {landmark_name}, Position: ({x}, {y})")  

    # Display the images with landmarks
    visualize_preprocessed_face(face_img, resized_landmarks) 