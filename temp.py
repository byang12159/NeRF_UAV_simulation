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

base = np.array( [[ 6.65833834e-01 , 1.66533454e-16, -7.46100064e-01],
 [-7.46100064e-01 , 5.55111512e-17, -6.65833834e-01 ],
 [-5.55111512e-17 , 1.00000000e+00,  1.66533454e-16]])
roat = R.from_matrix(base)
print("BASE",roat.as_euler('xyz', degrees=True))

comp = np.array( [[ 0.53549892 ,-0.12476546,  0.83526911],
 [-0.84291429, -0.14021918 , 0.51945556 ],
 [ 0.05231064, -0.98222816, -0.18025381  ]])
roat = R.from_matrix(comp)
print("comp",roat.as_euler('xyz', degrees=True))
# YAW ........ -0.8841654663472056

# FINAL C2W ...........
#  [[ 6.33935139e-01  1.11022302e-16 -7.73386216e-01 -7.53054405e-01]
#  [-7.73386216e-01  5.55111512e-17 -6.33935139e-01  1.41289163e-02]
#  [ 0.00000000e+00  1.00000000e+00  1.66533454e-16 -2.76663710e-01]]