
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


camera_path = 'camera_path.json'
with open(camera_path, 'r') as file:
    data = json.load(file)

cam_data = data.get('camera_path')
cam_states = np.zeros((len(cam_data),16))

for j in range(len(cam_data)):
    cam_states[j] = cam_data[j].get('camera_to_world')

print(cam_states.shape)

x = []
y = []
z = []

for i in range(cam_states.shape[0]):
    x.append(cam_states[i][3])
    y.append(cam_states[i][7])
    z.append(cam_states[i][11])
    
print(x)
# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(x, y, z, c='b', marker='o')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # Set axis limits
# ax.set_xlim([-2, 2])
# ax.set_ylim([-2, 2])
# ax.set_zlim([-2, 2])

# # Show the 3D plot
# plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample data
# x = [1, 2, 3, 4, 5]
# y = [5, 4, 3, 2, 1]
# z = [2, 3, 1, 5, 4]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c='b', marker='o')  # Scatter points

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
