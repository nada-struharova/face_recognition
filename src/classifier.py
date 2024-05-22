import tensorflow as tf
import numpy as np

# ----------------- LOAD MODEL ----------------- 
# Option 1: ENTIRE MODEL
loaded_model = tf.keras.models.load_model('face_recognition/src/global_features/model')

# Option 2: LOAD WEIGHTS TO EXTRACT FEATURES
loaded_model.load_weights('face_recognition/src/global_features/weights/fine_tuned_weights.h5') 

# ----------------- EXTRACTOR ----------------- 
# Create a new model that outputs the penultimate layer
feature_extractor = tf.keras.Model(inputs=loaded_model.input, outputs=loaded_model.layers[-2].output)  

# ----------------- EXTRACT GLOBAL FEATURES ----------------- 

def extract_global_features(image):
    # Load and Preprocess Image
    # img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))

    image = tf.keras.applications.vgg16.preprocess_input(image)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  

    # Extract features
    features = feature_extractor.predict(img_array)
    return features.flatten()  # Flatten to a 1D vector

def train_classifier(dataset_path):
    # Load images and identities
    images, identities = utils.load_dataset(dataset_path)
    all_features = []
    all_labels = []

    for image, identity in zip(images, identities):
        local_features = extract_local_features(image, identity)
        global_features = extract_global_features(image)  

        # Flatten local features into single vector (if needed)
        local_features = np.array(local_features[identity]).flatten()

        # Feature Fusion (e.g., concatenation)
        fused_features = np.concatenate([local_features, global_features])
        all_features.append(fused_features)
        all_labels.append(identity)

    # Train your classifier (e.g., Random Forest, SVM, etc.)
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)  # Choose your classifier
    clf.fit(X_train, y_train)
    
    # Evaluate 
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of classifier on local and global features: {accuracy:.2f}")
    return accuracy