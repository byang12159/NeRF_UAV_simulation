import json
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R

rot = np.array([[1,2,3],
                [4,5,6],
                [6,7,8]])

pos = np.array([[11],[12],[13]])

transformation_matrix = np.eye(4)  # Initialize as identity matrix
transformation_matrix[:3, :3] = rot  # Set the upper-left 3x3 submatrix as the rotation matrix
transformation_matrix[:3, 3] = pos  # Set the rightmost column as the translation vector

print(transformation_matrix)