import json
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
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

x = [1,2,3]
z = np.zeros(100)

print(z[:len(x)])