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

A = np.array([[ 6.33935139e-01 , 1.11022302e-16, -7.73386216e-01 ],
              [-7.73386216e-01,  5.55111512e-17, -6.33935139e-01 ],
              [ 0.00000000e+00,  1.00000000e+00  ,1.66533454e-16 ]])
roat = R.from_matrix(A)

print(roat.as_euler('xyz'))


# YAW ........ -0.8841654663472056

# FINAL C2W ...........
#  [[ 6.33935139e-01  1.11022302e-16 -7.73386216e-01 -7.53054405e-01]
#  [-7.73386216e-01  5.55111512e-17 -6.33935139e-01  1.41289163e-02]
#  [ 0.00000000e+00  1.00000000e+00  1.66533454e-16 -2.76663710e-01]]