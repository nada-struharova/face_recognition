import os
import cv2

def grayscale_images(directory):
  """
  This function reads images from a directory, converts them to grayscale, and saves them back to the same directory.

  Args:
      directory (str): The path to the directory containing the images.
  """

  for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for common image formats
      img_path = os.path.join(directory, filename)
      img = cv2.imread(img_path)
      gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      cv2.imwrite(img_path, gray_img)  # Overwrite the original image with the grayscale version

if __name__ == "__main__":
  image_directory = "face_recognition/datasets/grayscale_imgs"  # Replace with the actual path to your image directory
  grayscale_images(image_directory)