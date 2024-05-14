import numpy as np
import tensorflow as tf

def occlude_eyes(image):
    """ Add a rectangle occlusion over the eyes in the image to simulate sunglasses. 

    Args:
        image (tf.Tensor): The input image tensor.

    Returns:
        tf.Tensor: The occluded image.
    """
    h, _, _ = image.shape
    # Adding a horizontal bar across the eyes
    y_start, y_end = int(h * 0.3), int(h * 0.4)
    image[y_start:y_end, :, :] = 0  # Black bar
    return image
    
def occlude_rectangle(image, min_coverage=0.05, max_coverage=0.1, use_colour=False):
    """ Add a rectangle occlusion to original image.
    
    Args:
        image (tf.Tensor): The input image tensor.
        min_coverage (float): Minimum fraction of the image to cover.
        max_coverage (float): Maximum fraction of the image to cover.
        use_colour (bool): If True, use a random solid colour. Default is black.

    Returns:
        tf.Tensor: The occluded image.
    """

    # Get image dimensions (Image Shape = (batch_size, 224, 224, 3))
    # if len(image.shape) == 3:
    #     height, width, channels = image.shape
    # else:
    #     _, height, width, channels = image.shape
    # area = height * width

    height = 224
    width = 224
    channels = 3
    
    # Occlusion coverage and dimensions
    coverage = tf.random.uniform([], min_coverage, max_coverage)
    area = tf.cast(height * width, tf.float32)
    occlusion_area = tf.cast(area * coverage, tf.int32)
    occlusion_height = tf.cast(tf.sqrt(tf.cast(occlusion_area * height / width, tf.float32)), tf.int32)
    occlusion_width = occlusion_area // occlusion_height

    # Select colour
    if use_colour:
        colour = tf.random.uniform([3], 0, 255, dtype=tf.int32)
        colour = tf.cast(colour, image.dtype)
    else:
        colour = tf.zeros([3], dtype=image.dtype)

    upper_left_y = tf.random.uniform([], 0, height - occlusion_height, dtype=tf.int32)
    upper_left_x = tf.random.uniform([], 0, width - occlusion_width, dtype=tf.int32)

    mask = tf.pad(tensor=tf.ones([occlusion_height, occlusion_width, channels], dtype=image.dtype),
                  paddings=[[upper_left_y, height - occlusion_height - upper_left_y],
                            [upper_left_x, width - occlusion_width - upper_left_x],
                            [0, 0]],
                  mode='CONSTANT',
                  constant_values=0)
    
    return tf.where(mask == 1, colour, image)

def occlude_ellipse(image, min_coverage=0.05, max_coverage=0.1, use_colour=False):
    """ Add occlusions of various shapes to the image.
    
    Args:
        image (tf.Tensor): The input image tensor.
        min_coverage (float): Minimum fraction of the image to cover.
        max_coverage (float): Maximum fraction of the image to cover.
        use_colour (bool): If True, use a random solid colour. Default is black.

    Returns:
        tf.Tensor: The occluded image.
    """
    # Image dimensions
    # if len(image.shape) == 3:
    #     height, width, _ = image.shape
    # else:
    #     _, height, width, _ = image.shape

    height = 224
    width = 224
    channels = 3

    coverage = tf.random.uniform([], min_coverage, max_coverage)
    area = height * width
    occlusion_area = int(area * coverage)
    occlusion_height = int(tf.sqrt(occlusion_area * height / width))
    occlusion_width = int(occlusion_area / occlusion_height)

    if use_colour:
        colour = tf.random.uniform([3], 0, 255, dtype=tf.int32)
        colour = tf.cast(colour, tf.uint8)
    else:
        colour = tf.zeros([3], dtype=tf.uint8)

    center_x = tf.random.uniform([], occlusion_width // 2, width - occlusion_width // 2, dtype=tf.int32)
    center_y = tf.random.uniform([], occlusion_height // 2, height - occlusion_height // 2, dtype=tf.int32)

    rows, cols = tf.meshgrid(tf.range(height), tf.range(width), indexing='ij')
    mask = tf.square((rows - center_y) / occlusion_height) + tf.square((cols - center_x) / occlusion_width) <= 1
    mask = tf.stack([mask] * 3, axis=-1)

    return tf.where(tf.cast(mask, tf.bool), colour, image)

def occlude(image):
    """ Add either a rectangle or ellipse occlusion randomly. """
    if np.random.random() > 0.5:
        return occlude_rectangle(image, use_colour=True)
    else:
        return occlude_ellipse(image, use_colour=True)
    
# Augmentation dataset images with occlusion
def add_occlusion(label, image):
    image = occlude(image)  # Augment the image
    return label, image
