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
# from nerf_image import Nerf_image

from scipy.spatial.transform import Rotation as R

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# part of this script is adapted from iNeRF https://github.com/salykovaa/inerf
# and NeRF-Pytorch https://github.com/yenchenlin/nerf-pytorch/blob/master/load_llff.py

class NeRF:
    
    def __init__(self):
        #  def __init__(self, nerf_params):
        # Parameters
        # self.output_dir = './output/'
        # self.data_dir = nerf_params['data_dir']
        # self.model_name = nerf_params['model_name']
        # self.obs_img_num = nerf_params['obs_img_num']
        self.batch_size = 32 # number of pixels to use for measurement points
        self.factor = 4 # image down-sample factor
        self.focal = 635
        self.H = 720
        self.W = 1280
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
        
        

        self.focal = self.focal / self.factor
        self.H =  self.H / self.factor
        self.W =  self.W / self.factor
        self.H, self.W = int(self.H), int(self.W)

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

        self.coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, self.W - 1, self.W), np.linspace(0, self.H - 1, self.H)), -1),
                            dtype=int)

        if sampling_type == 'random':
            self.coords = self.coords.reshape(self.H * self.W, 2)

    def get_loss(self, particles, batch, base_img):
        target_s = self.obs_img_noised[batch[:, 1], batch[:, 0]] # TODO check ordering here
        target_s = torch.Tensor(target_s).to(device)
        losses = []

        start_time = time.time()

        for i, particle in enumerate(particles):
            #TODO: Potentially speed up render process by only generating specific rays, check locNerf
            yaw = 0
            compare_img = Nerf_image.render_Nerf_image(yaw)
            compare_img_points = compare_img[batch[:,0],batch[:,1]]
            compare_tensor = torch.tensor(compare_img_points)

            base_img_points = base_img[batch[:,0],batch[:,1]]
            base_tensor = torch.tensor(base_img_points)

            loss = img2mse(base_tensor,compare_tensor)
            losses.append(loss.item())
      
        nerf_time = time.time() - start_time
                   
        return losses, nerf_time
    
    def visualize_nerf_image(self, nerf_pose):
        pose_dummy = torch.from_numpy(nerf_pose).cuda()
        with torch.no_grad():
            print(nerf_pose)
            rgb, disp, acc, _ = render(self.H, self.W, self.focal, chunk=self.chunk, c2w=pose_dummy[:3, :4], **self.render_kwargs)
            rgb = rgb.cpu().detach().numpy()
            rgb8 = to8b(rgb)
            ref = to8b(self.obs_img)
        plt.imshow(rgb8)
        plt.show()