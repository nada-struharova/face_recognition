import numpy as np

def get_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def within_bbox(point, bbox):
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max

def get_z_score(sample, mean, std):
    return (sample - mean) / std

def filter_landmarks(face_data):
    """ Detect unreliable landmarks and disregard or de-emphasise impact in aggregated features. """
    landmarks = face_data['landmarks']

    # Example distances
    distances = {
        'eye_to_eye': get_distance(landmarks['right_eye'], landmarks['left_eye']),
        'eye_to_nose': get_distance(landmarks['right_eye'], landmarks['nose']),
        # More distances can be added here
    }

    # OUTLIER CHECK
    # TODO: get statistical data from normal non-occluded faces
    # Placeholder: These should be computed from a larger sample dataset
    expected_means = {'eye_to_eye': 120, 'eye_to_nose': 80}  # Hypothetical values
    expected_stds = {'eye_to_eye': 10, 'eye_to_nose': 8}  # Hypothetical values

    outliers = {}
    for key, distance in distances.items():
        z_score = get_z_score(distance, expected_means[key], expected_stds[key])
        if np.abs(z_score) > 2:  # Threshold for considering a value as an outlier
            outliers[key] = f"Outlier detected with z-score {z_score}"

    # POSITIONAL CHECKS
    for landmark, coords in landmarks.items():
        if not within_bbox(coords, face_data['facial_area']):
            landmarks[landmark] = None

    if landmarks['left_eye'][0] < landmarks['right_eye'][0]:
        landmarks['right_eye'] = None
        landmarks['left_eye'] = None
        # issues['eye_position'] = "Left eye is not on the left of the right eye"

    # Check if eyes are above nose
    if landmarks['right_eye'][1] > landmarks['nose'][1] or landmarks['left_eye'][1] > landmarks['nose'][1]:
        landmarks['right_eye'] = None
        landmarks['left_eye'] = None
    
    # Check if nose above mouth
    if landmarks['nose'][1] > landmarks['mouth_left'][1] or landmarks['nose'][1] > landmarks['mouth_right'][1]:
        landmarks['nose'] = None

    return face_data

def detect_occlusion(face_data):
    landmarks = face_data['landmarks']
    facial_area = face_data['facial_area']  # [x1, y1, x2, y2]

    # Bounding box dimensions
    bbox_width = facial_area[2] - facial_area[0]
    bbox_height = facial_area[3] - facial_area[1]

    # Landmarks
    right_eye = np.array(landmarks['right_eye'])
    left_eye = np.array(landmarks['left_eye'])
    nose = np.array(landmarks['nose'])
    # mouth_right = np.array(landmarks['mouth_right'])
    # mouth_left = np.array(landmarks['mouth_left'])

    # Define expected distances as proportions of the bounding box dimensions
    expected_eye_to_eye = get_distance(right_eye, left_eye)
    expected_eye_to_nose_right = 0.35 * bbox_width  # Adjusted proportion of the bounding box width
    expected_eye_to_nose_left = 0.35 * bbox_width  # Adjusted proportion of the bounding box width

    tolerance = 0.2  # 20% tolerance

    # Calculate actual distances
    actual_eye_to_nose_right = get_distance(right_eye, nose)
    actual_eye_to_nose_left = get_distance(left_eye, nose)

    # Determine occlusion
    occlusions = {}
    if not (1-tolerance) * expected_eye_to_nose_right < actual_eye_to_nose_right < (1+tolerance) * expected_eye_to_nose_right:
        occlusions['right_eye_to_nose'] = 'Occluded or misaligned'
    if not (1-tolerance) * expected_eye_to_nose_left < actual_eye_to_nose_left < (1+tolerance) * expected_eye_to_nose_left:
        occlusions['left_eye_to_nose'] = 'Occluded or misaligned'

    return occlusions

# Example data
face_data = {
    "landmarks": {
      "right_eye": [257, 210],
      "left_eye": [375, 252],
      "nose": [303, 300],
      "mouth_right": [228, 339],
      "mouth_left": [320, 375]
    }
}

occlusions = detect_occlusion(face_data)
print("Detected Occlusions:", occlusions)
