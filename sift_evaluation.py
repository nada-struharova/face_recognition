import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# Function to extract SIFT descriptors
def extract_sift_descriptors(image):
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

# Function to load images from a directory
def load_images_from_directory(directory):
    images = []
    labels = []
    label_names = os.listdir(directory)
    for label in label_names:
        image_files = os.listdir(os.path.join(directory, label))
        for image_file in image_files:
            image_path = os.path.join(directory, label, image_file)
            image = cv2.imread(image_path)
            images.append(image)
            labels.append(label)
    return images, labels

# Function to compute and plot intra-class and inter-class distances
def plot_distances(descriptors, labels, n_clusters=10):
    # Cluster descriptors using K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(descriptors)
    cluster_labels = kmeans.labels_

    # Calculate pairwise distances
    distances = pairwise_distances(descriptors, metric='euclidean')
    
    # Separate intra-class and inter-class distances
    intra_class_distances = []
    inter_class_distances = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                intra_class_distances.append(distances[i][j])
            else:
                inter_class_distances.append(distances[i][j])

    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.hist(intra_class_distances, bins=50, alpha=0.5, label='Intra-class distances')
    plt.hist(inter_class_distances, bins=50, alpha=0.5, label='Inter-class distances')
    plt.title('Intra-class vs Inter-class Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Main script
if __name__ == '__main__':
    # Load images and labels
    directory = 'face_recognition/datasets/face_bank'
    images, labels = load_images_from_directory(directory)

    # Extract SIFT descriptors
    all_descriptors = []
    image_labels = []
    for image, label in zip(images, labels):
        descriptors = extract_sift_descriptors(image)
        if descriptors is not None:
            all_descriptors.extend(descriptors)
            image_labels.extend([label] * len(descriptors))

    all_descriptors = np.array(all_descriptors)

    # Plot intra-class and inter-class distances
    plot_distances(all_descriptors, image_labels, n_clusters=10)
