import numpy as np
import cv2
import time

import matplotlib.pyplot as plt
import time
import json
from mpl_toolkits.mplot3d import Axes3D
from particle_filter import ParticleFilter
from utils import get_pose
from full_filter import NeRF
from nerf_image import Nerf_image
from controller import Controller
from scipy.spatial.transform import Rotation as R
import os
import torch 
class Run():
    def __init__(self, camera_path, nerf_file_path, width=320, height=320, fov=50):

        # self.nerfimage = Nerf_image(nerf_file_path)
        self.nerf = NeRF(nerf_file_path, width, height, fov)

        ####################### Import camera path trajectory json #######################
        with open(camera_path, 'r') as file:
            data = json.load(file)

        cam_data = data.get('camera_path')
        self.cam_states = np.zeros((len(cam_data),16))

        for j in range(len(cam_data)):
            self.cam_states[j] = cam_data[j].get('camera_to_world')

        # self.nerfW = data.get('render_height')
        # self.nerfH = data.get('render_width')
        self.nerfFov = (data.get('keyframes')[0].get('fov')) #Assuming all cameras rendered using same FOV 

        print("Finish importing camera states")
        self.nerfW = 320
        self.nerfH = 320
        ####################### Initialize Variables #######################

        # bounds for particle initialization, meters + degrees
        self.min_bounds = {'px':-0.5,'py':-0.5,'pz':0.0,'rz':-2.5,'ry':-179.0,'rx':-2.5}
        self.max_bounds = {'px':0.5,'py':0.5,'pz':0.5,'rz':2.5,'ry':179.0,'rx':2.5}

        self.num_particles = 300
        
        self.obs_img_pose = None
        self.center_about_true_pose = False
        self.all_pose_est = []
        
        self.rgb_input_count = 0

        self.use_convergence_protection = True
        self.convergence_noise = 0.2

        self.number_convergence_particles = 10 #number of particles to distribute

        self.sampling_strategy = 'random'
        self.photometric_loss = 'rgb'
        self.num_updates =0
        self.control = Controller()
        

        self.view_debug_image_iteration = 0 #view NeRF rendered image at estimated pose after number of iterations (set to 0 to disable)

        ####################### Generate Initial Particles #######################
        self.get_initial_distribution()

        # add initial pose estimate before 1st update step
        position_est = self.filter.compute_weighted_position_average()
        rot_est = self.filter.compute_simple_rotation_average()
        pose_est = np.eye(4)  # Initialize as identity matrix
        pose_est[:3, :3] = rot_est  # Set the upper-left 3x3 submatrix as the rotation matrix
        pose_est[:3, 3] = position_est  # Set the rightmost column as the translation vector
        self.all_pose_est.append(pose_est)
        
    def mat3d(self, x,y,z):
        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(x,y,z,'*')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(-5, 30)  # Set X-axis limits
        ax.set_ylim(-5, 30)  # Set Y-axis limits
        ax.set_zlim(-5, 30)  # Set Z-axis limits
        # Show the plot
        plt.show()

    def get_initial_distribution(self):
        # NOTE for now assuming everything stays in NeRF coordinates (x right, y up, z inward)

        # get distribution of particles from user, generate np.array of (num_particles, 6)
        self.initial_particles_noise = np.random.uniform(np.array(
            [self.min_bounds['px'], self.min_bounds['py'], self.min_bounds['pz'], self.min_bounds['rz'], self.min_bounds['ry'], self.min_bounds['rx']]),
            np.array([self.max_bounds['px'], self.max_bounds['py'], self.max_bounds['pz'], self.max_bounds['rz'], self.max_bounds['ry'], self.max_bounds['rx']]),
            size = (self.num_particles, 6))
        

        # Dict of position + rotation, with position as np.array(300x6)
        self.initial_particles = self.set_initial_particles()
        # self.mat3d(self.initial_particles.get('position')[:,0],self.initial_particles.get('position')[:,1],self.initial_particles.get('position')[:,2])

        # Initiailize particle filter class with inital particles
        self.filter = ParticleFilter(self.initial_particles)


    def set_initial_particles(self):
        initial_positions = np.zeros((self.num_particles, 3))
        rots = []
        for index, particle in enumerate(self.initial_particles_noise):
            # Initialize at origin location
            i = self.cam_states[0]
            # future = self.cam_states[1]
            # f_x = future[3]
            # f_y = future[7]
            
            # yaw = np.arctan2( f_y - i[7],f_x - i[3]  ) - np.pi/2
            
            # camera_to_world = np.array(i[:-4]).reshape((3,4))
            # rpy = R.from_euler('xyz', [np.deg2rad(90), 0, yaw])
            # camera_to_world[:,:-1] = rpy.as_matrix()
            # x = camera_to_world[0][3]
            # y = camera_to_world[1][3]
            # z = camera_to_world[2][3]
            x = i[3]
            y = i[7]
            z = i[11]
            rot1 = i.reshape(4,4)
            rot = rot1[:3,:3]
            gt_rotation_obj  = R.from_matrix(rot)
            gt_euler  =  gt_rotation_obj.as_euler('xyz')
            # phi = gt_euler[0]
            # theta = gt_euler[1]
            # psi = gt_euler[2]
            phi = gt_euler[0]
            theta = gt_euler[1]
            psi = gt_euler[2]

            # print("initialzia \n",rot1)
            # x = particle[0]
            # y = particle[1]
            # z = particle[2]
            # phi = particle[3]
            # theta = particle[4]
            # psi = particle[5]

            # print(x,y,z)
            # particle_pose = get_pose(phi, theta, psi, x, y, z, self.obs_img_pose, self.center_about_true_pose)
            
            # set positions
            initial_positions[index,:] = [x,y, z]

            # set orientations, create rotation object
            rots.append(gt_rotation_obj)
            # rots.append(R.from_matrix(particle_pose[0:3,0:3]))
            
            # print(initial_particles)

        # print("INITIAL POSITION ", initial_positions)
        # print("INITIAL ROT", rots)
        return {'position':initial_positions, 'rotation':np.array(rots)}

    

    def move(self, x0=np.zeros(12), goal=np.zeros(12), dt=0.1):
        # integrate dynamics
        movement = self.control.simulate(x0, goal, dt)
        pass

    def publish_pose_est(self, pose_est, img_timestamp = None):
        print("Pose Est",pose_est.shape)
        pose_est = self.move()
 
        position_est = pose_est[:3, 3]
        rot_est = R.as_quat(pose_est[:3, :3])

        # populate msg with pose information
        pose_est.pose.pose.position.x = position_est[0]
        pose_est.pose.pose.position.y = position_est[1]
        pose_est.pose.pose.position.z = position_est[2]
        pose_est.pose.pose.orientation.w = rot_est[0]
        pose_est.pose.pose.orientation.x = rot_est[1]
        pose_est.pose.pose.orientation.y = rot_est[2]
        pose_est.pose.pose.orientation.z = rot_est[3]
        # print(pose_est_gtsam.rotation().ypr())

        # publish pose
        self.pose_pub.publish(pose_est)

    def odometry_update(self,state0, state1):
        state_difference = state1-state0
        rot0 = R.from_matrix(state0[:3,:3])
        rot1 = R.from_matrix(state1[:3,:3])
        eul0 = rot0.as_euler('xyz')
        eul1 = rot1.as_euler('xyz')
        diffeul = eul1-eul0
        for i in range(self.num_particles):
            self.filter.particles['position'][i] += [state_difference[0][3], state_difference[1][3], state_difference[2][3]]

            peul = self.filter.particles['rotation'][i].as_euler('xyz')
            peul += diffeul
            prot = R.from_euler('xyz',peul)
            self.filter.particles['rotation'][i] = prot
        
        print("Finish odometry update")

    def rgb_run(self,iter, img,msg=None, get_rays_fn=None, render_full_image=False):
        print("processing image")
        start_time = time.time()
        self.rgb_input_count += 1

        # make copies to prevent mutations
        particles_position_before_update = np.copy(self.filter.particles['position'])
        particles_rotation_before_update = [i.as_matrix() for i in self.filter.particles['rotation']]

        # resize input image so it matches the scale that NeRF expects
        img = cv2.resize(img, (int(self.nerfW), int(self.nerfH)))
        self.nerf.obs_img = img
        show_true = self.view_debug_image_iteration != 0 and self.num_updates == self.view_debug_image_iteration-1
        #Create a grid to sample image points for comparison
        self.nerf.get_poi_interest_regions(show_true, self.sampling_strategy)
        # plt.imshow(self.nerf.obs_img)
        # plt.show()

        total_nerf_time = 0

        # if self.sampling_strategy == 'random':
        # From the meshgrid of image, find Batch# of points to randomly sample and compare, list of 2d coordinates
        rand_inds = np.random.choice(self.nerf.coords.shape[0], size=self.nerf.batch_size, replace=False)
        batch = self.nerf.coords[rand_inds]

        loss_poses = []
        for index, particle in enumerate(particles_position_before_update):
            if index %10 ==0:
                # print(f"PART STATE for iteration {iter}:   ",particles_position_before_update[index])
                pobj = R.from_matrix(particles_rotation_before_update[index])
                print(f"PART EULER for iteration {iter}:   \n",pobj.as_euler('xyz', degrees=True))
            loss_pose = np.zeros((4,4))
            rot = particles_rotation_before_update[index]
            loss_pose[0:3, 0:3] = rot
            loss_pose[0:3,3] = particle[0:3]
            loss_pose[3,3] = 1.0
            loss_poses.append(loss_pose)

        losses, nerf_time = self.nerf.get_loss(loss_poses, batch, img, iter=iter)
        print("Pass losses")
        print(losses)
        temp = 1
        for index, particle in enumerate(particles_position_before_update):
            self.filter.weights[index] = 1/(losses[index]+temp)

        total_nerf_time += nerf_time

        # Resample Weights
        self.filter.update()
        self.num_updates += 1
        
        position_est = self.filter.compute_weighted_position_average()
        rot_est = self.filter.compute_simple_rotation_average()
        pose_est = np.eye(4)  # Initialize as identity matrix
        pose_est[:3, :3] = rot_est  # Set the upper-left 3x3 submatrix as the rotation matrix
        pose_est[:3, 3] = position_est  # Set the rightmost column as the translation vector
        self.all_pose_est.append(pose_est)
        
        # Update odometry step
        current_state = self.cam_states[iter].reshape(4,4)
        next_state = self.cam_states[iter+1].reshape(4,4)
        self.odometry_update(current_state,next_state)
        # self.publish_pose_est(pose_est)


        update_time = time.time() - start_time
        print("forward passes took:", total_nerf_time, "out of total", update_time, "for update step")

        return pose_est

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
from scipy.spatial.transform import Rotation
import copy
import cv2

if __name__ == "__main__":

    camera_path = 'camera_path.json'
    nerf_file_path = './outputs/IRL1/nerfacto/2023-09-15_031235/config.yml'

    mcl = Run(camera_path,nerf_file_path, 80, 80, 50)      

 
    # Initialize Drone Position
    est_states = np.zeros((len(mcl.cam_states) ,3))
    gt_states  = np.zeros((len(mcl.cam_states) ,16))
    est_euler  = np.zeros((len(mcl.cam_states) ,3))  
    gt_euler   = np.zeros((len(mcl.cam_states) ,3))  
    iteration_count = np.arange(0,len(mcl.cam_states) , 1, dtype=int)
    
    # Assume constant time step between trajectory stepping
    for iter in range(len(mcl.cam_states)-1):
        new_dir_name = f"NeRF_UAV_simulation/images/Iteration_{iter}"
        if not os.path.exists(new_dir_name):
            os.mkdir(new_dir_name)

        base_img = mcl.nerf.render_Nerf_image_simple(mcl.cam_states[iter],mcl.cam_states[iter+1],save=False, save_name = "base", iter=iter, particle_number=None)
        # cv2.imshow("img ",base_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        brot = mcl.cam_states[iter].reshape(4,4)
        bobj = R.from_matrix(brot[:3,:3])
        print(f"BASE EULER for iteration {iter}:   \n",bobj.as_euler('xyz', degrees=True))
        
        pose_est = mcl.rgb_run(iter, base_img)   

        # Visualization
        est_states[iter] = pose_est[0:3,3].flatten()
        gt_states[iter] = mcl.cam_states[iter]

        est_rotation_obj = R.from_matrix(pose_est[0:3,0:3])
        est_euler[iter] = est_rotation_obj.as_euler('xyz', degrees=True)

        gt_matrix = gt_states[iter].reshape(4,4)
        gt_rotation_obj  = R.from_matrix(gt_matrix[0:3,0:3])
        gt_euler[iter]  =  gt_rotation_obj.as_euler('xyz', degrees=True)
           
        print(">>>>>>>>>>>>>> ")
        print(iteration_count[:iter+1])
        # print([np.linalg.norm(gt_states[:iter+1,3]-est_states[:iter+1,0], axis =1)])
        print(np.abs(gt_states[:iter+1,3]-est_states[:iter+1,0]))
        print(">>>>>>>>>>>>>> ")
    
        # Create a figure with six subplots (2 rows, 3 columns)
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 3, 1)
        plt.plot(iteration_count[:iter+1], np.abs(gt_states[:iter+1,3]-est_states[:iter+1,0]))
        plt.title('X error')

        plt.subplot(2, 3, 2)
        plt.plot(iteration_count[:iter+1], np.abs( gt_states[:iter+1,7]-est_states[:iter+1,1]) )
        plt.title('Y error')

        # Plot the third graph in the third subplot
        plt.subplot(2, 3, 3)
        plt.plot(iteration_count[:iter+1],np.abs(gt_states[:iter+1,11]-est_states[:iter+1,2]) )
        plt.title('Z error')

        # Plot the fourth graph in the fourth subplot
        plt.subplot(2, 3, 4)
        plt.plot(iteration_count[:iter+1],np.abs(est_euler[:iter+1,0]-gt_euler[:iter+1,0]) )
        plt.title('Yaw error')

        # Plot the fifth graph in the fifth subplot
        plt.subplot(2, 3, 5)
        plt.plot(iteration_count[:iter+1],np.abs(est_euler[:iter+1,1]-gt_euler[:iter+1,1]))
        plt.title('Pitch error')

        # Plot the sixth graph in the sixth subplot
        plt.subplot(2, 3, 6)
        plt.plot(iteration_count[:iter+1],np.abs(est_euler[:iter+1,2]-gt_euler[:iter+1,2]))
        plt.title('Roll error')

        plt.tight_layout()
        file_path = f'./NeRF_UAV_simulation/Plots/plot{iter}.png'
        plt.savefig(file_path)
        plt.close()
        
        print('cam_states[iter]',mcl.cam_states[iter])
        print('pose est',pose_est)

    print("########################Done########################")