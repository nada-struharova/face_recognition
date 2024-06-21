import numpy as np
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

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

    print("finished loading dataset")
    return images, identities

# ------------------ Image Preprocessing ------------------ 
def preprocess_face(image, bbox, landmarks):
    """Preprocesses a face image using MTCNN face detected data.

    Args:
        image: The original image.
        bbox: Facial area (x, y, width, height).
        landmarks: Dictionary of facial landmarks.

    Returns:
        The preprocessed (aligned, resized) face image and updated landmarks.
    """
    x, y, width, height = bbox

    # 1. Cropping with Margins: 
    margin_factor = 0.1  # Adjust as needed
    margin_x = int(width * margin_factor)
    margin_y = int(height * margin_factor)
    
    # Ensure cropping stays within image bounds
    x_start = max(0, x - margin_x)
    y_start = max(0, y - margin_y)
    x_end = min(image.shape[1], x + width + margin_x)
    y_end = min(image.shape[0], y + height + margin_y)

    face_img = image[y_start:y_end, x_start:x_end]

    # 2. Update Landmarks After Cropping
    new_landmarks = {
        landmark_name: (lx - x_start, ly - y_start)
        for landmark_name, (lx, ly) in landmarks.items()
    }

    # 3. Alignment Based on Eyes
    left_eye, right_eye = new_landmarks['left_eye'], new_landmarks['right_eye']
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))  
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # 4. Rotate Cropped Image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_img = cv2.warpAffine(
        face_img, rotation_matrix, face_img.shape[1::-1], flags=cv2.INTER_LINEAR
    )

    # 5. Update Landmarks After Rotation
    for landmark_name, (lx, ly) in new_landmarks.items():
        pt = np.array([lx, ly, 1])
        rotated_pt = rotation_matrix.dot(pt)
        new_landmarks[landmark_name] = (int(rotated_pt[0]), int(rotated_pt[1]))

    # 6. Normalize and Resize 
    output_size = (224, 224)
    resized_img = cv2.resize(aligned_img, output_size)

    # 7. Update Landmarks After Normalization and Resize
    scale_x = output_size[0] / aligned_img.shape[1]
    scale_y = output_size[1] / aligned_img.shape[0]
    new_landmarks = {
        key: (int(value[0] * scale_x), int(value[1] * scale_y))
        for key, value in new_landmarks.items()
    }
    
    return resized_img, new_landmarks

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

def split_to_regions(image, landmarks, region_size=(80, 80)):
    """
    Split a (224, 224) face image into fixed-size regions based on landmark coordinates.
    The regions prioritize including the specific landmark within the region boundaries.

    Args:
    - image: The (224, 224) face image. Can be a NumPy array or TensorFlow tensor.
    - landmarks: Dictionary containing landmark coordinates.
    - region_size: Tuple indicating the size of each region (height, width).

    Returns:
    - A list of region images.
    """

    region_height, region_width = region_size

    def get_region(x, y):
        # Calculate region boundaries
        left = x - region_width // 2
        top = y - region_height // 2

        # Ensure the region is within image bounds
        if left < 0:
            left = 0
        elif left + region_width > 224:
            left = 224 - region_width

        if top < 0:
            top = 0
        elif top + region_height > 224:
            top = 224 - region_height

        right = left + region_width
        bottom = top + region_height

        # Check if the image is a tensor or a NumPy array
        if isinstance(image, tf.Tensor):
            region = image[top:bottom, left:right, :]
        else:
            region = image[top:bottom, left:right]

        return region

    # Extract regions based on landmarks
    regions = [
        get_region(*landmarks['left_eye']),
        get_region(*landmarks['right_eye']),
        get_region(*landmarks['nose']),
        get_region(*landmarks['mouth_left']),
        get_region(*landmarks['mouth_right'])
    ]

    return regions

def split_tensor_to_grid(image, grid_size=(2, 2)):
    """
    Split an image into a grid of fixed-size regions.
    
    Args:
    - image: The input image tensor of shape (height, width, channels).
    - grid_size: Tuple indicating the number of rows and columns in the grid.
    
    Returns:
    - A list of region tensors.
    """
    # Ensure image is a TensorFlow tensor
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image)

    height, width = tf.shape(image)[0], tf.shape(image)[1]
    grid_rows, grid_cols = grid_size
    row_size = height // grid_rows
    col_size = width // grid_cols 

    regions = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            # Calculate coordinates for the region
            x1, y1 = j * col_size, i * row_size
            x2, y2 = x1 + col_size, y1 + row_size
            
            # Extract the region from the image tensor
            region = image[y1:y2, x1:x2, :]

            # Ensure the region tensor has the expected dtype
            if region.dtype != tf.uint8:
                region = tf.cast(region, tf.uint8)
                
            regions.append(region)

    return regions

def visualize_regions(image, regions, landmarks, grid=False, grid_size=(6,6)):
    """
    Visualize the image with landmarks and the corresponding regions.

    Args:
    - image: The image as a NumPy array or TensorFlow tensor.
    - regions: List of region images.
    - landmarks_dict: Dictionary containing landmark coordinates.

    Returns:
    - None (displays the visualization).
    """
    if not grid:
        # Plot the image and regions
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 3, 1)
        plt.imshow(image[...,::-1])# plt.imshow(image.numpy().astype('uint8'))
        plt.title('Preprocessed Image')
        plt.axis('off')

        # Plot the landmarks on the original image
        for landmark in landmarks.values():
            plt.plot(landmark[0], landmark[1], color='lime', marker='.')

        for i, region in enumerate(regions):
            plt.subplot(2, 3, i + 2)
            plt.imshow(region[...,::-1]) #plt.imshow(region.numpy().astype('uint8'))
            plt.title(list(landmarks.keys())[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()
    if grid:
        # Plot Grid
        plt.figure(figsize=(6, 6))
        for idx, region in enumerate(regions):
            plt.subplot(grid_size[0], grid_size[1], idx + 1)
            plt.imshow(region[...,::-1])
            plt.axis('off')
        plt.tight_layout()
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

# def discriminative_power(true_labels, decision_scores):
#     """
#     Args:
#         true_labels: Pairs of images and labels (1 for intra-identity, 0 for otherwise).
#         decision_scores: Calculated distances.
#     """
#     fpr, tpr, thresholds = roc_curve(true_labels, decision_scores)
#     roc_auc = auc(fpr, tpr)

#     plt.figure()
#     plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], linestyle='--') # Random guess line
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc="lower right")
#     plt.show()