import json
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R

import os

# Get the current working directory
# current_directory = os.getcwd()
# iter = 1
# print(current_directory)
# base_img = cv2.imread("./images/foo{}.png".format(iter))
# print("./images/foo{}.png".format(iter))
# print(base_img.shape)
# cv2.imshow("base",base_img)
# cv2.waitKey(0)

current_directory = os.getcwd()
iter = 1
print(current_directory)
# base_img = cv2.imread("C:/Users/byang/Downloads/nerfstudio/NeRF_UAV_simulation/images/foo12.png")
base_img = cv2.imread("./NeRF_UAV_simulation/images/foo12.png")
print(base_img.shape)
cv2.imshow("base",base_img)
cv2.waitKey(0)