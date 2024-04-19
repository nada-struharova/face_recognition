## COMPARE MAX AND AVERAGE POOLING FOR FACE MATCHING AND RECOGNITION
# 1. Implement both max and average pooling
# 2. Use a standard dataset including cases of partial occlusion and pose variation (e.g. LWF)
# 3. Measure accuracy, recall, precision, F1-score of face matching using each pooling method
# 4. Analyse results -> which pooling method performs best for face matching in case
    # of occlusion and pose variation

import numpy as np
from scipy.spatial.distance import euclidean

def max_pooling(features, pool_size):
    """ Apply max pooling to the feature array. """
    num_features, feature_length = features.shape
    pooled_features = np.empty((num_features, feature_length // pool_size))

    for i in range(num_features):
        for j in range(0, feature_length, pool_size):
            pooled_features[i, j // pool_size] = np.max(features[i, j:j + pool_size])

    return pooled_features

def average_pooling(features, pool_size):
    """ Apply average pooling to the feature array. """
    num_features, feature_length = features.shape
    pooled_features = np.empty((num_features, feature_length // pool_size))

    for i in range(num_features):
        for j in range(0, feature_length, pool_size):
            pooled_features[i, j // pool_size] = np.mean(features[i, j:j + pool_size])

    return pooled_features

def generate_features(num_samples, feature_length):
    """ Generate synthetic features for a number of samples. """
    return np.random.rand(num_samples, feature_length)

def compare_features(features1, features2):
    """ Compare two sets of features using Euclidean distance. """
    distances = [euclidean(features1[i], features2[i]) for i in range(len(features1))]
    return distances

# Parameters
num_faces = 10
feature_length = 100
pool_size = 5

# Generate synthetic features
features = generate_features(num_faces, feature_length)

# Apply pooling
max_pooled_features = max_pooling(features, pool_size)
avg_pooled_features = average_pooling(features, pool_size)

# Simulate comparing two sets of features (as if matching faces)
max_distances = compare_features(max_pooled_features, max_pooled_features)  # Self-comparison
avg_distances = compare_features(avg_pooled_features, avg_pooled_features)  # Self-comparison

print("Max Pooling Distances:\n", max_distances)
print("Average Pooling Distances:\n", avg_distances)

# Optionally, compare max and average pooled features between different synthetic "faces"
# This step simulates matching different faces
inter_max_distances = compare_features(max_pooled_features[:-1], max_pooled_features[1:])
inter_avg_distances = compare_features(avg_pooled_features[:-1], avg_pooled_features[1:])

print("Inter-Face Max Pooling Distances:\n", inter_max_distances)
print("Inter-Face Average Pooling Distances:\n", inter_avg_distances)
