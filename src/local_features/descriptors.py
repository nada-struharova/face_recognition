import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import daisy
from skimage.feature import rgb2gray

# ------------------ Baseline Features ------------------ 
def extract_raw_pixel_features(image):
    """
    Extracts raw pixel intensities from an image. Assumes the image is grayscale.

    Args:
        image: A NumPy array representing the image.

    Returns:
        A flattened NumPy array of pixel intensities.
    """
    return image.flatten()

def extract_random_features(image):
    """
    Generates random features of the same dimensionality as an image.

    Args:
        image: A NumPy array representing the image (used to determine dimensionality).

    Returns:
        A NumPy array of random features. 
    """
    num_features = image.size  # Total number of pixels
    return np.random.rand(num_features)

# ------------------ Local Feature Descriptors ------------------ 
def extract_sift_features(region):
    """Extract SIFT features from an image region.
    Args:
        image_region: A grayscale image region

    Returns:
        Numpy array containing flattened SIFT features. 
    """

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(region, None)

    print("SIFT keypoints detected:", len(keypoints))

    if descriptors is None or len(keypoints) < 1:  # Handle exceptions
        descriptors = np.zeros((1, 128), dtype=np.float32)  # Default SIFT descriptor size

    # Apply max-pooling (you can change this to average pooling if desired)
    pooled_features = np.max(descriptors, axis=0)

    return pooled_features

def extract_surf_features(region, hessianThreshold=400):
    """ Extract SURF features from an face region.
    Args:
        image_region: A grayscale image region

    Returns:
        Numpy array containing flattened SURF features.
    """

    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
    _, descriptors = surf.detectAndCompute(region, None)

    if descriptors is None: # Handle exceptions
        descriptors = np.zeros((1, surf.descriptorSize()), dtype=np.float32)

    return descriptors.flatten() # Flatten for concatenation 

def extract_hog_features(region, win=(64, 64), block=(16,16), stride=(4,4), cell=(4,4), bins=9):
    """ Extract HOG features from an face region.
    Args:
        image_region: A grayscale image region

    Returns:
        Numpy array containing flattened SURF features.
    """
    hog = cv2.HOGDescriptor(win, block, stride, cell, bins)
    hog_features = hog.compute(region)
    return hog_features.flatten().ravel() # Flatten for concatenation 

def extract_lbp_features(region, radius=3, num_points=8):
    """Extract LBP features from a face region.
    Args:
        image_region: A grayscale image region
        radius: Radius for the LBP circular neighborhood
        num_points: Number of points in LBP circular neighborhood
    Returns:
        Numpy array containing LBP histogram (feature vector)
    """
    num_points = num_points * radius
    lbp = local_binary_pattern(region, num_points, radius, method='uniform') 
    n_bins = int(lbp.max() + 1)  # Uniform LBP has a fixed range of values
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    print("Dimension of LBP descriptor:", hist.shape)
    return hist

def extract_daisy_descriptors(image, step=1, radius=15, rings=2, histograms=2, orientations=8, normalization='l1', visualize=False):
    """
    Extracts DAISY descriptors from a grayscale image.

    Args:
        image: The input grayscale image.
        step: The sampling step size.
        radius: The radius of the outermost ring.
        rings: Number of rings.
        histograms: Number of histograms per ring.
        orientations: Number of orientations per histogram.
        normalization: The descriptor normalization method ('l1' or 'l2').
        visualize: If True, visualize the descriptors.

    Returns:
        descriptors: A numpy array of DAISY descriptors.
        keypoints: A numpy array of keypoint coordinates (if visualize=True).
    """
    # Error Handling
    # If image is RGB, convert to grayscale
    if len(image.shape) == 3:
        image = rgb2gray(image) 

    # Extract DAISY descriptors
    descriptors = daisy(
        image, step=step, radius=radius, rings=rings, histograms=histograms,
        orientations=orientations, normalization=normalization
    )

    # Visualize descriptors (optional)
    if visualize:
        from skimage.feature import draw_keypoints
        import matplotlib.pyplot as plt

        keypoints = np.array([
            (i, j) for i in range(0, image.shape[0], step)
            for j in range(0, image.shape[1], step)
        ])
        plt.imshow(draw_keypoints(image, keypoints, descriptors))
        plt.show()

    return descriptors

def extract_orb_descriptors(image, num_keypoints=1000):
    """
    Extracts ORB keypoints and descriptors from a grayscale image.

    Args:
        image: The input grayscale image.
        num_keypoints: The maximum number of keypoints to detect.

    Returns:
        keypoints: A list of KeyPoint objects.
        descriptors: A numpy array of ORB descriptors.
    """
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=num_keypoints)  # Adjust num_keypoints if needed

    # Find the keypoints and descriptors with ORB
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors
def extract_orb_keypoints_sift_descriptors(image, num_keypoints=1000):
    """
    Extracts ORB keypoints and SIFT descriptors from a grayscale image.

    Args:
        image: The input grayscale image.
        num_keypoints: The maximum number of keypoints to detect.

    Returns:
        keypoints: A list of KeyPoint objects.
        descriptors: A numpy array of SIFT descriptors.
    """

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=num_keypoints)

    # Find the keypoints with ORB
    orb_keypoints = orb.detect(image, None)

    # Convert keypoints to KeyPoint objects for SIFT
    orb_keypoints = [cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size) for kp in orb_keypoints]

    # Initialize SIFT descriptor extractor
    sift = cv2.SIFT_create()  

    # Compute SIFT descriptors based on the ORB keypoints
    orb_keypoints, sift_descriptors = sift.compute(image, orb_keypoints)  

    return orb_keypoints, sift_descriptors