import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the LFW dataset
dataset, dataset_info = tfds.load('lfw', split='train', with_info=True, as_supervised=False)

# Print dataset information
print(dataset_info)

# Iterate over the dataset to show images and their labels
for example in dataset.take(5):  # Display only the first 5 examples
    image, label = example['image'], example['label']
    plt.figure()
    plt.imshow(image.numpy())
    plt.title(label.numpy().decode('utf-8'))
    plt.show()