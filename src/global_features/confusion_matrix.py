import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, num_classes):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data
        self.num_classes = num_classes
    
    def on_epoch_end(self, epoch, logs=None):
        val_pred_probs = self.model.predict(self.validation_data.map(lambda x, y: x))  # Predict probabilities on validation data
        
        # Extract one-hot encoded labels from the dataset
        val_labels = np.concatenate([y.numpy() for x, y in self.validation_data])
        
        # Convert predicted probabilities to class labels
        val_pred_labels = np.argmax(val_pred_probs, axis=1)
        val_true_labels = np.argmax(val_labels, axis=1)
        
        # Compute metrics
        val_accuracy = logs['val_accuracy']
        val_precision = precision_score(val_true_labels, val_pred_labels, average='weighted', zero_division=1)
        val_recall = recall_score(val_true_labels, val_pred_labels, average='weighted', zero_division=1)
        val_f1 = f1_score(val_true_labels, val_pred_labels, average='weighted')
        
        # Print metrics
        print("Epoch:", epoch + 1)
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        print(f'Validation Precision: {val_precision:.4f}')
        print(f'Validation Recall: {val_recall:.4f}')
        print(f'Validation F1-score: {val_f1:.4f}')

        # Add metrics to logs
        logs['val_precision'] = val_precision
        logs['val_recall'] = val_recall
        logs['val_f1'] = val_f1

# class Precision(tf.keras.metrics.Metric):
#     def __init__(self, name='precision', **kwargs):
#         super(Precision, self).__init__(name=name, **kwargs)
#         self.true_positives = self.add_weight(name='tp', initializer='zeros')
#         self.false_positives = self.add_weight(name='fp', initializer='zeros')

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.argmax(y_pred, axis=-1)
#         y_true = tf.cast(y_true, tf.int32)
#         y_pred = tf.cast(y_pred, tf.int32)
#         true_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
#         false_positives = tf.reduce_sum(tf.cast(tf.not_equal(y_true, y_pred), tf.float32))
#         self.true_positives.assign_add(true_positives)
#         self.false_positives.assign_add(false_positives)

#     def result(self):
#         precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
#         return precision

#     def reset_states(self):
#         self.true_positives.assign(0.0)
#         self.false_positives.assign(0.0)
        
# class Recall(tf.keras.metrics.Metric):
#     def __init__(self, name='recall', **kwargs):
#         super(Recall, self).__init__(name=name, **kwargs)
#         self.true_positives = self.add_weight(name='tp', initializer='zeros')
#         self.false_negatives = self.add_weight(name='fn', initializer='zeros')

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.argmax(y_pred, axis=-1)
#         y_true = tf.cast(y_true, tf.int32)
#         y_pred = tf.cast(y_pred, tf.int32)
#         true_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
#         false_negatives = tf.reduce_sum(tf.cast(tf.not_equal(y_true, y_pred), tf.float32))
#         self.true_positives.assign_add(true_positives)
#         self.false_negatives.assign_add(false_negatives)

#     def result(self):
#         recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
#         return recall

#     def reset_states(self):
#         self.true_positives.assign(0.0)
#         self.false_negatives.assign(0.0)

# class F1Score(tf.keras.metrics.Metric):
#     def __init__(self, name='f1_score', **kwargs):
#         super(F1Score, self).__init__(name=name, **kwargs)
#         self.precision = Precision()
#         self.recall = Recall()

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         self.precision.update_state(y_true, y_pred, sample_weight)
#         self.recall.update_state(y_true, y_pred, sample_weight)

#     def result(self):
#         precision = self.precision.result()
#         recall = self.recall.result()
#         f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
#         return f1_score

#     def reset_states(self):
#         self.precision.reset_states()
#         self.recall.reset_states()