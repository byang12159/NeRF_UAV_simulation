# Script to generate test images for particle filter debugging 

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

from full_filter import NeRF

new_dir_name = f"NeRF_UAV_simulation/debugging_images"
if not os.path.exists(new_dir_name):
    os.mkdir(new_dir_name)

target_pose = [ 0.7270040306017818, 1.6653345369377348e-16, -0.686633191368407, -0.654864857277347, 
               -0.686633191368407, 5.551115123125783e-17, -0.7270040306017818, 0.10130739443730398, 
               -5.551115123125783e-17, 1.0, 1.6653345369377348e-16, -0.2069397511450955,
                0.0, 0.0, 0.0, 1.0
            ]

nerf_file_path = './outputs/IRL1/nerfacto/2023-09-15_031235/config.yml'
nerf = NeRF(nerf_file_path, width=320, height=320, fov=50)


target_pose = np.array(target_pose).reshape(4,4)

cam2world = np.zeros((3,4))
cam2world[:3,:3] = target_pose[:3,:3]
cam2world[0:3,3] = target_pose[0:3,3]

base_img = nerf.render_Nerf_image_debug(cam2world, save=False, save_name='particle', iter=0,particle_number=0)

# cv2.imshow("IMG",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
def compute_orb(img, name):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img * 255).astype(np.uint8)
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
     
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(gray_image,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(gray_image, kp)
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(gray_image, kp, None, color=(0,255,0), flags=0)
    plt.imshow(img2), plt.show()

    output_dir = f"NeRF_UAV_simulation/debugging_images/Translated/translated_{name}.jpg"
    cv2.imwrite(output_dir, img2)

    return kp

# Generate Tests
################################# translation shift #################################
pose_translated = []
for x in range(-5,5,2):
    for y in range(-5,5,2):
        for z in range(-5,5,2):
            translation = np.zeros((3,4))
            translation[0,3] = x/10.
            translation[1,3] = y/10.
            translation[2,3] = z/10.
            pose_translated.append(np.copy(cam2world) + translation)

print("Checkpoint1")
new_dir_name = f"NeRF_UAV_simulation/debugging_images/Translated"
if not os.path.exists(new_dir_name):
    os.mkdir(new_dir_name)

kp_base = compute_orb(base_img, 'base')
base_len = len(kp_base)
minimum_point = int(base_len*0.3) #Minimum required points for comparison, else ignore image
for index, translated in enumerate(pose_translated):
    img = nerf.render_Nerf_image_debug(translated, save=False, save_name='particle', iter=0,particle_number=0)
    kp = compute_orb(img, name = index)
    print("len kp",len(kp))

    if len(kp) < minimum_point:
        # loss = max
        continue
    else:
        sample_size = min(base_len, len(kp))
        
    print(sample_size)

    print(index)

################################# rotation shift #################################
################################# combined rotation + translate shift #################################


# img = cv2.imread('NeRF_UAV_simulation/debugging_images/Translated/translated_118.jpg', cv2.IMREAD_GRAYSCALE)

print("DONE")