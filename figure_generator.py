
import torch 
import numpy as np 
import json 
import os 
from pathlib import Path
import yaml
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imgaug.augmenters as iaa

def set_fog_properties(img, val):
    aug = iaa.Fog(
        intensity_mean = (255, 255),
        density_multiplier = (val, val)
    )
    image_fog = aug(image = img)
    return image_fog

def set_dark_properties(img, val):
    aug = iaa.Add(val*100)
    image_dark = aug(image=img)
    return image_dark

base = cv2.imread("Demo_image.jpg")
print("BASE",base[0])
base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
# from full_filter import set_dark_properties, set_fog_properties

# f = set_fog_properties(base,1)
# cv2.imshow()

import cv2
import numpy as np

# Load or create your 9 images (replace 'image1' to 'image9' with your image data)
image1 = set_dark_properties(set_fog_properties(base,1), -1) #Row1
image2 = set_dark_properties(set_fog_properties(base,1), 0)#Row1
image3 = set_dark_properties(set_fog_properties(base,1), 1)#Row1
image4 = set_dark_properties(set_fog_properties(base,0.5),-1 )#Row2
image5 = set_dark_properties(set_fog_properties(base,0.5), 0)#Row2
image6 = set_dark_properties(set_fog_properties(base,0.5), 1)#Row2
image7 = set_dark_properties(set_fog_properties(base,0), -1)#Row3
image8 = set_dark_properties(set_fog_properties(base,0), 0)#Row3
image9 = set_dark_properties(set_fog_properties(base,0), 1)#Row3
# Repeat this for images 3 to 9...

cv2.imwrite("IRL1.jpg",image1)
cv2.imwrite("IRL2.jpg",image2)
cv2.imwrite("IRL3.jpg",image3)
cv2.imwrite("IRL4.jpg",image4)
cv2.imwrite("IRL5.jpg",image5)
cv2.imwrite("IRL6.jpg",image6)
cv2.imwrite("IRL7.jpg",image7)
cv2.imwrite("IRL8.jpg",image8)
cv2.imwrite("IRL9.jpg",image9)
# # Check if all images were loaded successfully
# if all(image is not None for image in [image1, image2, image3, image4, image5, image6, image7, image8, image9]):
#     # Get the dimensions of one of the images (assuming all have the same dimensions)
#     height, width, _ = image1.shape

#     # Create an empty grid image with a white background
#     grid_image = np.ones((3 * height, 3 * width, 3), dtype=np.uint8) * 255

#     # Place each image in the grid
#     grid_image[0:height, 0:width] = image1
#     grid_image[0:height, width:2*width] = image2
#     grid_image[0:height, 2*width:3*width] = image3
#     grid_image[height:2*height, 0:width] = image4
#     grid_image[height:2*height, width:2*width] = image5
#     grid_image[height:2*height, 2*width:3*width] = image6
#     grid_image[2*height:3*height, 0:width] = image7
#     grid_image[2*height:3*height, width:2*width] = image8
#     grid_image[2*height:3*height, 2*width:3*width] = image9

#     # Show the grid image
#     cv2.imshow('Grid of Images', grid_image)

#     # Wait for a key press and close the display window
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Failed to load one or more of the images.")