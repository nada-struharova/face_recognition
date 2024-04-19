import numpy as np

def concatenate_features(*features):
    """ Concatenate local features into a single descriptor.
        - flatten feature arrays into 1D per sample. """
    concatenated_features = np.hstack([feat.reshape(feat.shape[0], -1) for feat in features if feat is not None])
    return concatenated_features

def normalise_features(features):
    """ Normalize each feature vector using L2 norm. """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / np.maximum(norms, 1e-8)
    return normalized_features

    # norms = np.linalg.norm(features, axis=1, keepdims=True)
    # normalized_features = features / np.maximum(norms, 1e-8)  # Using max to avoid division by zero
    # return normalized_features

def integrate_local_global_features(local_features, global_features):
    """ Combine local and global features into a unified descriptor.
        - feature sets have to be flattened and concatenated per sample. """
    # Ensure both feature arrays are numpy arrays
    local_features = np.array(local_features)
    global_features = np.array(global_features)
    
    # Concatenate local and global features
    combined_features = np.hstack([local_features, global_features])
    return combined_features