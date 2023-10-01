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

class Nerf_image:
    def __init__(self, path):

        script_dir = os.path.dirname(os.path.realpath(__file__))

        # config_fn = os.path.join('./nerf_env/nerf_env/outputs/IRL2/nerfacto/2023-09-21_210511/config.yml')
        config_fn = os.path.join(self.path)
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
        
    def render_Nerf_image(self, camera_to_world):

            camera = Cameras(camera_to_worlds = camera_to_world, fx = self.fx, fy = self.fy, cx = self.cx, cy = self.cy, width=self.width, height=self.height, camera_type=self.camera_type)
            camera = camera.to('cuda')
            ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)

            with torch.no_grad():
                tmp = model.get_outputs_for_camera_ray_bundle(ray_bundle)

            img = tmp['rgb']
            img =(colormaps.apply_colormap(image=img, colormap_options=colormaps.ColormapOptions())).cpu().numpy()

            return img


if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # config_fn = os.path.join('./nerf_env/nerf_env/outputs/IRL2/nerfacto/2023-09-21_210511/config.yml')
    config_fn = os.path.join('./outputs/IRL1/nerfacto/2023-09-15_031235/config.yml')
    config_path = Path(config_fn)
    _, pipeline, _, step = eval_setup(
        config_path,
        eval_num_rays_per_chunk=None,
        test_mode='inference'
    )

    model = pipeline.model
    
    f = open('camera_path.json')

    data = json.load(f)
    import matplotlib.image

    j  =0
    for idx, i  in enumerate(data['camera_path']):
        
        future = data['camera_path'][idx+1]
        f_x = future['camera_to_world'][3]
        f_y = future['camera_to_world'][7]
        
        yaw = np.arctan2( f_y - i['camera_to_world'][7],f_x - i['camera_to_world'][3]  ) - np.pi/2
        
        print("yaw:", yaw)
        
        
        camera_to_world = np.array(i['camera_to_world'][:-4]).reshape((3,4))
        camera_to_world[0,:-1] = [np.cos(yaw), -np.sin(yaw), 0]
        camera_to_world[1,:-1] = [np.sin(yaw), np.cos(yaw), 0]
        camera_to_world[2,:-1] = [0,0,1]
        
        pitch = np.array([[0,0,1],[0,1,0],[-1,0,0]])
        
        camera_to_world[:,:-1] = camera_to_world[:,:-1]@pitch

        camera_to_world = torch.FloatTensor([ camera_to_world ])
        
        fx = fy = (320.0/2)/(np.tan(np.deg2rad(50)/2))
        cx = cy = 160.0
        width = height = 320
        distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        camera_type  = CameraType.PERSPECTIVE
        camera = Cameras(camera_to_worlds = camera_to_world, fx = fx, fy = fy, cx = cx, cy = cy, width=width, height=height, camera_type=camera_type)
        camera = camera.to('cuda')
        ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)
        print(len(ray_bundle))


        # model = Model(config, scene_box, num_train_data=1)


        # loaded_state = torch.load(model_fn)
        # model.load_state_dict(loaded_state)
        with torch.no_grad():
            tmp = model.get_outputs_for_camera_ray_bundle(ray_bundle)

        # img = model.get_rgba_image(res)
        img = tmp['rgb']
        img =(colormaps.apply_colormap(image=img, colormap_options=colormaps.ColormapOptions())).cpu().numpy()
        print(img.shape)

        matplotlib.image.imsave('images/foo'+ str(j) + '.png', img)
        #plt.imshow(img)

        #plt.show()

        j+=1


        print("stop")