import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data
    
    def on_epoch_end(self, epoch, logs=None):
        val_ds = self.validation_data

        # Gather true labels and predictions
        val_labels = []
        val_preds = []
        for images, labels in val_ds:
            val_labels.append(labels.numpy())
            preds = self.model.predict(images)
            val_preds.append(np.argmax(preds, axis=1))

        val_labels = np.concatenate(val_labels)
        val_preds = np.concatenate(val_preds)
        
        # Convert one-hot encoded labels to integer labels
        val_true_labels = np.argmax(val_labels, axis=1)

        # Compute metrics
        accuracy = accuracy_score(val_true_labels, val_preds)
        precision = precision_score(val_true_labels, val_preds, average='weighted', zero_division=0)
        recall = recall_score(val_true_labels, val_preds, average='weighted', zero_division=0)
        f1 = f1_score(val_true_labels, val_preds, average='weighted', zero_division=0)

        # Print metrics
        print(f"\nEpoch: {epoch + 1}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1-score: {f1:.4f}\n")

        # Log metrics
        logs['val_accuracy'] = accuracy
        logs['val_precision'] = precision
        logs['val_recall'] = recall
        logs['val_f1'] = f1

class TripletMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_triplets):
        super(TripletMetricsCallback, self).__init__()
        self.validation_triplets = validation_triplets
    
    def on_epoch_end(self, epoch, logs=None):
        val_triplets = self.validation_triplets

        # Gather true labels and predictions for anchor, positive, and negative images
        anchor_labels = []
        positive_labels = []
        negative_labels = []
        for anchor_images, positive_images, negative_images in val_triplets:
            anchor_labels.append(self.model.predict(anchor_images))
            positive_labels.append(self.model.predict(positive_images))
            negative_labels.append(self.model.predict(negative_images))

        anchor_labels = np.concatenate(anchor_labels)
        positive_labels = np.concatenate(positive_labels)
        negative_labels = np.concatenate(negative_labels)

        # Compute embeddings for anchor, positive, and negative images
        anchor_embeddings = np.linalg.norm(anchor_labels, axis=1)
        positive_embeddings = np.linalg.norm(positive_labels, axis=1)
        negative_embeddings = np.linalg.norm(negative_labels, axis=1)

        # Calculate triplet loss and accuracy
        triplet_loss = np.mean(np.maximum(positive_embeddings - anchor_embeddings + 0.2, 0) +
                               np.maximum(anchor_embeddings - negative_embeddings + 0.2, 0))
        triplet_accuracy = np.mean(positive_embeddings < anchor_embeddings)  # Accuracy is when positive is closer than negative

        # Print metrics
        print(f"\nEpoch: {epoch + 1}")
        print(f"Triplet Loss: {triplet_loss:.4f}")
        print(f"Triplet Accuracy: {triplet_accuracy:.4f}")

        # Log metrics
        logs['triplet_loss'] = triplet_loss
        logs['triplet_accuracy'] = triplet_accuracy
