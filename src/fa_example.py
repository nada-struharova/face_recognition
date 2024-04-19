import face_alignment
from skimage import io
import cv2
import numpy as np
import torch

# Load Image
# TODO: Provide correct input image path
input_image = io.imread("face-alignment/test/assets/aflw-test.jpg")

# Initialise face alignment
fa_2d = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='mps')
fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='mps')
# Move all tensors to mps device via tensor.to(torch.device("mps"))

# Get 2D and 3D landmarks (68 in each category)
# If bounding boxes are needed -> kwarg return_bboxes=True
# TODO: try MediaPipe for higher number of landmarks (face mesh?)
preds_2d = fa_2d.get_landmarks_from_image(input_image)
preds_3d = fa_3d.get_landmarks_from_image(input_image)

# Print out and process further
print(preds_2d)
print(preds_3d)