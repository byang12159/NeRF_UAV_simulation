from matplotlib.markers import MarkerStyle
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from utils import show_img, find_POI, img2mse, load_llff_data, get_pose
from full_nerf_helpers import load_nerf
from render_helpers import render, to8b, get_rays
from particle_filter import ParticleFilter
from nerf_image import Nerf_image

from scipy.spatial.transform import Rotation as R

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
import matplotlib.image

# part of this script is adapted from iNeRF https://github.com/salykovaa/inerf
# and NeRF-Pytorch https://github.com/yenchenlin/nerf-pytorch/blob/master/load_llff.py

class NeRF:
    
    def __init__(self,path):
        #  def __init__(self, nerf_params):
        # Parameters
        # self.output_dir = './output/'
        # self.data_dir = nerf_params['data_dir']
        # self.model_name = nerf_params['model_name']
        # self.obs_img_num = nerf_params['obs_img_num']
        self.batch_size = 32 # number of pixels to use for measurement points
        self.factor = 4 # image down-sample factor

        # self.near = nerf_params['near']
        # self.far = nerf_params['far']
        # self.spherify = False
        # self.kernel_size = nerf_params['kernel_size']
        # self.lrate = nerf_params['lrate']
        # self.dataset_type = nerf_params['dataset_type']
        # self.sampling_strategy = nerf_params['sampling_strategy']
        self.delta_phi, self.delta_theta, self.delta_psi, self.delta_x, self.delta_y, self.delta_z = [0,0,0,0,0,0]
        # self.no_ndc = nerf_params['no_ndc']
        # self.dil_iter = nerf_params['dil_iter']
        self.chunk = 65536 # 1024x64 # number of rays processed in parallel, decrease if running out of memory
        # self.bd_factor = nerf_params['bd_factor']

        # print("dataset type:", self.dataset_type)
        # print("no ndc:", self.no_ndc)
        
        #Render 2d Images
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # config_fn = os.path.join('./nerf_env/nerf_env/outputs/IRL2/nerfacto/2023-09-21_210511/config.yml')
        config_fn = os.path.join(path)
        # config_fn = os.path.join(self.path)
        config_path = Path(config_fn)
        _, pipeline, _, step = eval_setup(
            config_path,
            eval_num_rays_per_chunk=None,
            test_mode='inference'
        )
        
        self.model = pipeline.model
        self.fx = (320.0/2)/(np.tan(np.deg2rad(50)/2))
        self.fy = (320.0/2)/(np.tan(np.deg2rad(50)/2))
        self.cx = 160.0
        self.cy = 160.0
        self.nerfW = 320
        self.nerfH = 320
        self.camera_type  = CameraType.PERSPECTIVE

        self.focal = self.fx
        # self.focal = self.focal / self.factor
        # self.nerfH =  self.nerfH / self.factor
        # self.nerfW =  self.nerfW / self.factor
        # self.nerfH, self.nerfW = int(self.nerfH), int(self.nerfW)

        # we don't actually use obs_img_pose when we run live images. this prevents attribute errors later in the code
        self.obs_img_pose = None

 
        # Load NeRF Model
        # self.render_kwargs = load_nerf(nerf_params, device)
        # bds_dict = {
        #     'near': self.near,
        #     'far': self.far,
        # }
        # self.render_kwargs.update(bds_dict)
    
    def get_poi_interest_regions(self, show_img=False, sampling_type = None):
        # TODO see if other image normalization routines are better
        self.obs_img_noised = (np.array(self.obs_img) / 255.0).astype(np.float32)

        if show_img:
            plt.imshow(self.obs_img_noised)
            plt.show()

        self.coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, self.nerfW - 1, self.nerfW), np.linspace(0, self.nerfH - 1, self.nerfH)), -1),
                            dtype=int)

        if sampling_type == 'random':
            self.coords = self.coords.reshape(self.nerfH * self.nerfW, 2)
    
    def render_Nerf_image(self, orientation: np.ndarray, position: np.ndarray, save, save_name, iter,particle_number):

        euler_angles_degrees = orientation.as_euler('xyz', degrees=True)

        rpy = R.from_euler('xyz', [np.deg2rad(90) + euler_angles_degrees[0], euler_angles_degrees[1], euler_angles_degrees[2]])

        camera_to_world = np.zeros((3,4))
        camera_to_world[:,-1] = position
        camera_to_world[:,:-1] = rpy.as_matrix()
        camera_to_world = torch.FloatTensor( camera_to_world )
        camera = Cameras(camera_to_worlds = camera_to_world, fx = self.fx, fy = self.fy, cx = self.cx, cy = self.cy, width=self.nerfW, height=self.nerfH, camera_type=self.camera_type)
        camera = camera.to('cuda')
        ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)

        with torch.no_grad():
            tmp = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)

        img = tmp['rgb']
        img =(colormaps.apply_colormap(image=img, colormap_options=colormaps.ColormapOptions())).cpu().numpy()

        if save:
            output_dir = f"NeRF_UAV_simulation/images/Iteration_{iter}/{save_name}{particle_number}.jpg"
            cv2.imwrite(output_dir, img)

        return img
    
    def render_Nerf_image_simple(self,state_now, state_future, save, save_name, iter,particle_number):
        i = state_now
        future = state_future
        f_x = future[3]
        f_y = future[7]
        
        yaw = np.arctan2( f_y - i[7],f_x - i[3]  ) - np.pi/2
        print("YAW ........",yaw)
        camera_to_world = np.array(i[:-4]).reshape((3,4))
        print("1c2w",camera_to_world)
        rpy = R.from_euler('xyz', [np.deg2rad(90), 0, yaw])
        print("rpy",rpy.as_matrix())
        camera_to_world[:,:-1] = rpy.as_matrix()
        print("FINAL C2W ...........\n",camera_to_world)
        camera_to_world = torch.FloatTensor( camera_to_world )

        camera = Cameras(camera_to_worlds = camera_to_world, fx = self.fx, fy = self.fy, cx = self.cx, cy = self.cy, width=self.nerfW, height=self.nerfH, camera_type=self.camera_type)
        camera = camera.to('cuda')
        ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)

        with torch.no_grad():
            tmp = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)

        img = tmp['rgb']
        img =(colormaps.apply_colormap(image=img, colormap_options=colormaps.ColormapOptions())).cpu().numpy()

        if save:
            output_dir = f"NeRF_UAV_simulation/images/Iteration_{iter}/{save_name}{particle_number}.jpg"
            cv2.imwrite(output_dir, img)

        return img
    def get_loss(self, particle_poses, batch, base_img, iter):
        target_s = self.obs_img_noised[batch[:, 1], batch[:, 0]] # TODO check ordering here
        target_s = torch.Tensor(target_s).to(device)
        losses = []

        start_time = time.time()

        for i, particle in enumerate(particle_poses):
            print(i)
            if i%10 == 0:
                compare_img = self.render_Nerf_image(R.from_matrix(particle[0:3,0:3]), particle[0:3,3],save=True, save_name='particle', iter=iter,particle_number=i)
            else:
                compare_img = self.render_Nerf_image(R.from_matrix(particle[0:3,0:3]), particle[0:3,3],save=False, save_name='particle', iter=iter,particle_number=i)
            compare_img_points = compare_img[batch[:,0],batch[:,1]]
            compare_tensor = torch.tensor(compare_img_points)

            base_img_points = base_img[batch[:,0],batch[:,1]]
            base_tensor = torch.tensor(base_img_points)

            # print("SIZE COMPARE",base_tensor.shape, compare_tensor.shape)
            loss = img2mse(base_tensor,compare_tensor)
            losses.append(loss.item())


      
        nerf_time = time.time() - start_time
                   
        return losses, nerf_time
    
    def visualize_nerf_image(self, nerf_pose):
        pose_dummy = torch.from_numpy(nerf_pose).cuda()
        with torch.no_grad():
            print(nerf_pose)
            rgb, disp, acc, _ = render(self.nerfH, self.nerfW, self.focal, chunk=self.chunk, c2w=pose_dummy[:3, :4], **self.render_kwargs)
            rgb = rgb.cpu().detach().numpy()
            rgb8 = to8b(rgb)
            ref = to8b(self.obs_img)
        plt.imshow(rgb8)
        plt.show()

    