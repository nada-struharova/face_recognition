import os
import cv2
import numpy as np
from mtcnn import MTCNN
from local_feature_extraction import LocalFeatureExtractor
import region_utils
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to load and preprocess images
def load_and_preprocess_images(image_dir):
    images = []
    identities = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg')):
                try:
                    img_path = os.path.join(root, file)
                    image = cv2.imread(img_path)
                    if image is not None:
                        images.append(image)
                        identities.append(file)
                    else:
                        print(f"Could not read image: {img_path}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    return images, identities

# Main function to process directory and create cluster diagram
def process_directory_and_cluster(image_dir):
    feature_extractor = LocalFeatureExtractor()
    images, identities = load_and_preprocess_images(image_dir)

    all_features = []
    valid_identities = []

    for image, identity in zip(images, identities):
        try:
            feat_dict = feature_extractor.extract_local_features(image, identity)
            if feat_dict.get(identity):
                all_features.append(feat_dict[identity])
                valid_identities.append(identity)
            else:
                print(f"Skipping image {identity} as no face was detected.")
        except Exception as e:
            print(f"Error extracting features from {identity}: {e}")

    all_features = np.array(all_features)
    kmeans = KMeans(n_clusters=5, random_state=42).fit(all_features)
    pca = PCA(n_components=2).fit_transform(all_features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca[:, 0], pca[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.legend(*scatter.legend_elements(), title='Cluster')
    for i, identity in enumerate(valid_identities):
        plt.annotate(identity, (pca[i, 0], pca[i, 1]))
    plt.title('Cluster Diagram of SIFT Feature Vectors')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

if __name__ == "__main__":
    image_dir = 'path_to_your_images_directory'
    process_directory_and_cluster(image_dir)