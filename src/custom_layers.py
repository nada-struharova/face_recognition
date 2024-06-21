import tensorflow as tf
import region_utils
from local_descriptors import DescriptorExtractorTF
from dataset_utils import preprocess_tf_input
import cv2
import numpy as np

class SIFTLayer(tf.keras.layers.Layer):
    def __init__(self, num_features=500, **kwargs):
        super(SIFTLayer, self).__init__(**kwargs)
        self.num_features = num_features
        self.sift = cv2.SIFT_create(nfeatures=self.num_features)

    def sift_descriptors(self, images):
        sift_descriptors = []
        for image in images:
            kp, des = self.sift.detectAndCompute(image.numpy(), None)
            if des is None:
                des = np.zeros((1, 128), dtype=np.float32)  # Handle images with no keypoints
            sift_descriptors.append(des)
        return np.array(sift_descriptors)

    def call(self, inputs):
        sift_features = tf.py_function(func=self.sift_descriptors, inp=[inputs], Tout=tf.float32)
        sift_features.set_shape((None, self.num_features, 128))
        return sift_features

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_features, 128)

class PreprocessVGG16(tf.keras.layers.Layer):
    def __init__(self, version=1, data_format=None, **kwargs):
        super(PreprocessVGG16, self).__init__(**kwargs)
        self.version = version
        self.data_format = data_format or tf.keras.backend.image_data_format()

    def call(self, inputs):
        return preprocess_tf_input(inputs, self.version, self.data_format)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(PreprocessVGG16, self).get_config()
        config.update({
            'version': self.version,
            'data_format': self.data_format
        })
        return config
    
class PreprocessGrayscale(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PreprocessGrayscale, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.image.rgb_to_grayscale(inputs)
    
# Custom layer for local feature extraction in global VGG16-based model
class LocalFeatureLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):  # Removed output_dim parameter
        super(LocalFeatureLayer, self).__init__(**kwargs)
        # self.descriptor_extractor = DescriptorExtractorTF()
        self.sift = cv2.SIFT_create()

        self.avg_landmarks = {'left_eye': (63, 85),
                              'mouth_left': (68, 170),
                              'mouth_right': (159, 171),
                              'nose': (113, 132),
                              'right_eye': (167, 85)}
        
    def call(self, inputs):
        fused_features = self.fuse_regions(inputs)
        return fused_features

    def fuse_regions(self, image, sift=True, hog=False, lbp=False):
        regions = region_utils.split_to_regions(image, self.avg_landmarks)
        fused_features = tf.constant([], dtype=tf.float32)

        for region in regions:
            # region_features = self.descriptor_extractor.get_region_features_tf(region, sift, hog, lbp)
            region_features = self.sift_descriptors(region)
            norm = tf.norm(region_features, ord='euclidean')
            
            # Use tf.cond to perform conditional check
            fused_features = tf.cond(tf.equal(norm, 0),
                                    lambda: self.handle_zero_norm(fused_features),
                                    lambda: fused_features / norm)

            fused_features = tf.concat([fused_features, region_features], axis=0)

        print("Final grid descriptor shape: ", fused_features.shape)
        return fused_features
    
    def sift_descriptors(self, region):
        region_uint8 = tf.image.convert_image_dtype(region, tf.uint8)
        kp, des = tf.py_function(self.detect_and_compute_sift_alt, [region_uint8], [tf.float32, tf.float32])
        if des is None:
            des = tf.zeros((1, 128), dtype=tf.float32)  # Handle regions with no keypoints
        return des

    def detect_and_compute_sift_alt(self, region_uint8):
        image = cv2.cvtColor(region_uint8.numpy(), cv2.COLOR_GRAY2RGB)
        kp, des = self.sift.detectAndCompute(image, None)
        return kp, des

# Create a custom layer to unpack triplets and apply base_model
class TripletEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, base_model):
        super(TripletEmbeddingLayer, self).__init__()
        self.base_model = base_model

    def call(self, inputs):
        anchor, positive, negative = tf.split(inputs, num_or_size_splits=3, axis=1)
        anchor_embedding = self.base_model(tf.squeeze(anchor, axis=1))  # Remove extra dim
        positive_embedding = self.base_model(tf.squeeze(positive, axis=1))
        negative_embedding = self.base_model(tf.squeeze(negative, axis=1))
        return tf.stack([anchor_embedding, positive_embedding, negative_embedding], axis=1)

class TripletUnpackLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        print(inputs)
        return tf.unstack(inputs, axis=1)
