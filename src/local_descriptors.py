import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import tensorflow as tf

class LocalDescriptorExtractor:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.hog = cv2.HOGDescriptor((64, 64), (16,16), (4,4), (4,4), 9)

    def get_feature_dim(self, sift=True, hog=False, lbp=False):
        dim_per_region = 0
        if sift:
            dim_per_region += 128  # SIFT dimension per region
        if hog:
            dim_per_region += 3780  # HOG dimensions
        if lbp:
            dim_per_region += 59   # Observed dimension for skimage LBP

        regions = len(self.avg_landmarks)
        return dim_per_region * regions

    # ------------------ Baseline Features ------------------ 
    def extract_raw_pixel_features(self, image):
        return image.flatten()

    def extract_random_features(self, image):
        num_features = image.size
        return np.random.rand(num_features)

    # ------------------ Local Feature Descriptors ------------------ 
    def extract_sift_features(self, region):
        keypoints, descriptors = self.sift.detectAndCompute(region, None)
        if descriptors is None or len(keypoints) < 1:
            descriptors = np.zeros((1, 128), dtype=np.float32)
        pooled_features = np.max(descriptors, axis=0)
        return pooled_features

    def extract_hog_features(self, region, win=(64, 64), block=(16, 16), stride=(4, 4), cell=(4, 4), bins=9):
        hog_features = self.hog.compute(region)
        return hog_features.flatten().ravel()

    def extract_lbp_features(self, region, radius=3, num_points=8):
        num_points = num_points * radius
        lbp = local_binary_pattern(region, num_points, radius, method='uniform') 
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        return hist

    def get_region_features(self, region, sift=True, hog=False, lbp=False):
        features = np.array([])
        if sift:
            sift_feat = self.extract_sift_features(region)
            if not isinstance(sift_feat, np.ndarray):
                sift_feat = np.array(sift_feat)
            features = np.concatenate([features, sift_feat])
        if hog:
            hog_feat = self.extract_hog_features(region)
            if not isinstance(hog_feat, np.ndarray):
                hog_feat = np.array(hog_feat)
            features = np.concatenate([features, hog_feat])
        if lbp:
            lbp_feat = self.extract_lbp_features(region)
            if not isinstance(lbp_feat, np.ndarray):
                lbp_feat = np.array(lbp_feat)
            features = np.concatenate([features, lbp_feat])
        return features
    
    def extract_sift_features_tf(self, region):
        # Convert image to uint8
        region_uint8 = tf.image.convert_image_dtype(region, tf.uint8)

        # Use OpenCV SIFT
        keypoints, descriptors = self.sift.detectAndCompute(region_uint8.numpy(), None)
        if descriptors is None or len(keypoints) < 1:
            descriptors = tf.zeros((1, 128), dtype=tf.float32)
        else:
            descriptors = tf.convert_to_tensor(descriptors, dtype=tf.float32)

        # Max-pooling
        pooled_features = tf.reduce_max(descriptors, axis=0)
        return pooled_features
    
    def extract_lbp_features_tf(self, region, radius=3, num_points=8):
        # Convert image to uint8
        region_uint8 = tf.image.convert_image_dtype(region, tf.uint8)

        # Compute LBP using TensorFlow
        lbp = tf.image.extract_patches(region_uint8, sizes=[1, radius*2, radius*2, 1], strides=[1, radius, radius, 1], rates=[1, 1, 1, 1], padding='VALID')
        lbp = tf.reshape(lbp, [-1, radius*2, radius*2])
        lbp = tf.map_fn(lambda patch: tf.py_function(self.compute_lbp_histogram, [patch.numpy(), num_points], tf.float32), lbp)
        hist = tf.reduce_mean(lbp, axis=0)

        return hist

    def compute_lbp_histogram_tf(self, patch, num_points):
        lbp = local_binary_pattern(patch, num_points, 1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=num_points+2, range=(0, num_points+2), density=True)
        return hist

    def get_region_features_tf(self, region, sift=True, hog=False, lbp=False):
        features = tf.constant([], dtype=tf.float32)
        if sift:
            sift_feat = self.extract_sift_features_tf(region)
            if not isinstance(sift_feat, tf.Tensor):
                sift_feat = tf.convert_to_tensor(sift_feat, dtype=tf.float32)
            features = tf.concat([features, sift_feat], axis=0)
        if hog:
            hog_feat = self.extract_hog_features_tf(region)
            if not isinstance(hog_feat, tf.Tensor):
                hog_feat = tf.convert_to_tensor(hog_feat, dtype=tf.float32)
            features = tf.concat([features, hog_feat], axis=0)
        if lbp:
            lbp_feat = self.extract_lbp_features(region)
            if not isinstance(lbp_feat, tf.Tensor):
                lbp_feat = tf.convert_to_tensor(lbp_feat, dtype=tf.float32)
            features = tf.concat([features, lbp_feat], axis=0)
        return features
    
class DescriptorExtractorTF:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def get_region_features_tf(self, region, sift=True, hog=True, lbp=True):
        features = tf.constant([], dtype=tf.float32)

        if sift:
            sift_feat = self.extract_sift_features(region)
            features = tf.concat([features, sift_feat], axis=0)

        if hog:
            hog_feat = self.extract_hog_features(region)
            features = tf.concat([features, hog_feat], axis=0)

        if lbp:
            lbp_feat = self.extract_lbp_features(region)
            features = tf.concat([features, lbp_feat], axis=0)

        return features

    def extract_sift_features(self, region):
        region_uint8 = tf.image.convert_image_dtype(region, tf.uint8)
        kp, des = tf.py_function(self.sift.detectAndCompute, [region_uint8, None], [tf.float32, tf.float32])
        if des is None:
            des = tf.zeros((1, 128), dtype=tf.float32)  # Handle regions with no keypoints
        return des

    def extract_hog_features(self, region):
        region_uint8 = tf.image.convert_image_dtype(region, tf.uint8)
        hog_features = tf.image.extract_patches(region_uint8, sizes=[1, 64, 64, 1], strides=[1, 4, 4, 1], rates=[1, 1, 1, 1], padding='VALID')
        hog_features = tf.reduce_mean(hog_features, axis=[1, 2, 3])
        return hog_features

    def extract_lbp_features(self, region, radius=3, num_points=8):
        region_uint8 = tf.image.convert_image_dtype(region, tf.uint8)
        lbp = tf.image.extract_patches(region_uint8, sizes=[1, radius*2, radius*2, 1], strides=[1, radius, radius, 1], rates=[1, 1, 1, 1], padding='VALID')
        lbp = tf.reshape(lbp, [-1, radius*2, radius*2])
        lbp = tf.map_fn(lambda patch: tf.py_function(self.compute_lbp_histogram, [patch.numpy(), num_points], tf.float32), lbp)
        hist = tf.reduce_mean(lbp, axis=0)
        return hist

    def compute_lbp_histogram(self, patch, num_points):
        lbp = local_binary_pattern(patch, num_points, 1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=num_points+2, range=(0, num_points+2), density=True)
        return hist