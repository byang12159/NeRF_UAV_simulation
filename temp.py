import nerfstudio
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox
import torch 
import numpy as np 
import json 
import os 
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
import yaml
import matplotlib.pyplot as plt 
from nerfstudio.utils import colormaps
from torchvision.utils import save_image
from scipy.spatial.transform import Rotation as R
import cv2

# def quatWAvgMarkley(Q, weights):
#     '''
#     Averaging Quaternions.

#     Arguments:
#         Q(ndarray): an Mx4 ndarray of quaternions.
#         weights(list): an M elements list, a weight for each quaternion.
#     '''

#     # Form the symmetric accumulator matrix
#     A = np.zeros((4, 4))
#     M = Q.shape[0]
#     wSum = 0

#     for i in range(M):
#         q = Q[i, :]
#         w_i = weights[i]
#         A += w_i * (np.outer(q, q)) # rank 1 update
#         wSum += w_i

#     # scale
#     A /= wSum

#     # Get the eigenvector corresponding to largest eigen value
#     return np.linalg.eigh(A)[1][:, -1]

# quaternion = np.array([[0.707, 0, 0.707, 0],
#                         [0.5, 0.5, 0.5, 0.5],
#                         [0.2, 0.51, 0.1, 0.5],
#                         [0.3, 0.2, 0.4, 0.52]])

# weights = [1,1,1,1]

# print(quatWAvgMarkley(quaternion, weights))

# print(np.linalg.norm(np.linalg.eigh(np.einsum('ij,ik,i->...jk', quaternion, quaternion, weights))[1][:, -1]))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion




def quaternion_difference(q1, q2):
    # Ensure both input quaternions are unit quaternions
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    
    # Calculate the relative quaternion (q2 * q1^-1)
    relative_quaternion = np.dot(q2, np.conjugate(q1))
    
    # Normalize the relative quaternion
    relative_quaternion /= np.linalg.norm(relative_quaternion)
    
    return relative_quaternion

def quaternion_addition(q1, q2):
    return q1 + q2

# Define two quaternions (w, x, y, z)

e1 = [0,0,0]
e2 = [20,0,5]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original and rotated vectors
ax.quiver(0, 0, 0, e1[0], e1[1], e1[2], color='b', label='Original Vector')
ax.quiver(0, 0, 0, e2[0], e2[1], e2[2], color='r', label='Rotated Vector')
# ax.quiver(0, 0, 0, p3[0], p3[1], p3[2], color='r', label='Rotated Vector')

# Set axis limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Show the plot
plt.show()




eobj1 = R.from_euler('xyz',e1)
eobj2 = R.from_euler('xyz',e2)
quaternion1 = eobj1.as_quat()
quaternion2 = eobj2.as_quat()
print("quaternion1",quaternion1)
print("quaternion2",quaternion2)
# Calculate the relative quaternion representing the difference between quaternion2 and quaternion1
relative_quaternion2 = quaternion_difference(quaternion1, quaternion2)
relative_quaternion = np.linalg.inv(quaternion1) * quaternion2  # Note that the order matters (q2 * q1.conjugate)

# Define a third quaternionq
q3 = Quaternion(1, 0, 0, 0)  # Identity quaternion (no rotation)
print("Relative Quaternion (Difference):", relative_quaternion)
r1 = relative_quaternion*quaternion2
relative_obj = R.from_quat(r1)
print("tranformed Euler (Difference):", relative_obj.as_euler('xyz'))

# Add the two quaternions
quaternion3 = quaternion_addition(quaternion1, relative_quaternion)


# Create a vector to be rotated
original_vector = np.array([1, 0, 0])

# # Apply the quaternion rotation to the vector
# rotated_vector = rotation_quaternion.rotate(original_vector)

r1 = R.from_quat(quaternion1)
r2 = R.from_quat(quaternion2)
r3 = R.from_quat(quaternion3)

m1 = r1.as_matrix()
m2 = r2.as_matrix()
m3 = r3.as_matrix()

p1 = np.dot(m1,original_vector)
p2 = np.dot(m2,original_vector)
p3 = np.dot(m3,original_vector)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original and rotated vectors
ax.quiver(0, 0, 0, p1[0], p1[1], p1[2], color='b', label='Original Vector')
ax.quiver(0, 0, 0, p2[0], p2[1], p2[2], color='r', label='Rotated Vector')
# ax.quiver(0, 0, 0, p3[0], p3[1], p3[2], color='r', label='Rotated Vector')

# Set axis limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Show the plot
plt.show()

print("p1",p1)
print("p2",p2)
print("p3",p3)