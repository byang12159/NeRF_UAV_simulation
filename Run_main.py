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
import pickle 
import copy 

class Run():
    def __init__(self, camera_path, nerf_file_path, width = 320, height = 320, fov = 50):

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
        self.nerfW = width
        self.nerfH = height
        ####################### Initialize Variables #######################

        initial_rotation = self.cam_states[0].reshape(4,4)
        print("initalrotaion",initial_rotation)
        initial_rotation_obj = R.from_matrix(initial_rotation[:3,:3])
        initial_rotation_eul = initial_rotation_obj.as_euler('xyz')
        self.initialization_center = [self.cam_states[0][3],self.cam_states[0][7],self.cam_states[0][11],initial_rotation_eul[0],initial_rotation_eul[1],initial_rotation_eul[2]]
        # bounds for particle initialization, meters + degrees
        # Isseus pitch and roll
        # self.min_bounds = {'px':-1.0,'py':-1.0,'pz':-1.0,'rz':-1.5,'ry':-1.5,'rx':-1.5}
        # self.max_bounds = {'px':1.0,'py':1.0,'pz':1.0,'rz':1.5,'ry':1.5,'rx':1.5}

        # Y 14, P13, R25
        # self.min_bounds = {'px':-1.0,'py':-1.0,'pz':-1.0,'rz':-0.5,'ry':-0.5,'rx':-0.5}
        # self.max_bounds = {'px':1.0,'py':1.0,'pz':1.0,'rz':0.5,'ry':0.5,'rx':0.5}

        # Good              
        # self.min_bounds = {'px':-1.0,'py':-1.0,'pz':-1.0,'rz':-0.2,'ry':-0.2,'rx':-0.2}
        # self.max_bounds = {'px':1.0,'py':1.0,'pz':1.0,'rz':0.2,'ry':0.2,'rx':0.2}

        self.min_bounds = {'px':-0.01,'py':-0.01,'pz':-0.01,'rz':-0.0,'ry':-0.0,'rx':-0.0}
        self.max_bounds = {'px':0.01,'py':0.01,'pz':0.01,'rz':0.0,'ry': 0.0,'rx': 0.0}

        self.min_bounds_odometry = {'px':-0.004,'py':-0.004,'pz':-0.004,'rz':-0.001,'ry':-0.001,'rx':-0.001}
        self.max_bounds_odometry = {'px':0.004,'py':0.004,'pz':0.004,'rz':0.001,'ry': 0.001,'rx': 0.001}

        self.num_particles = 100
        
        self.obs_img_pose = None
        self.center_about_true_pose = False
        self.all_pose_est = []


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
        position_est = self.filter.compute_simple_position_average()
        rot_est = self.filter.compute_simple_rotation_average()
        pose_est = np.zeros(3+4)  # Initialize as identity matrix
        pose_est[:3] = position_est 
        pose_est[3:] = rot_est
        self.all_pose_est.append(pose_est)

        self.last_state = None
        
    def center_euler(self, euler_angles):
        # Ensure the differences are within the range of -pi to pi
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        pitch_diff = (pitch_diff + np.pi) % (2 * np.pi) - np.pi
        roll_diff = (roll_diff + np.pi) % (2 * np.pi) - np.pi

        return [yaw_diff, pitch_diff, roll_diff]

    def mat3d(self):
        traj = np.zeros((len(self.cam_states),3))
        for i in range(len(self.cam_states)):
            traj[i][0] = self.cam_states[i][4]
            traj[i][1] = self.cam_states[i][7]
            traj[i][2] = self.cam_states[i][11]

        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(traj[:,0],traj[:,1],traj[:,2],'*')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(-5, 5)  # Set X-axis limits
        ax.set_ylim(-5, 5)  # Set Y-axis limits
        ax.set_zlim(-5, 5)  # Set Z-axis limits
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
        
        # Initiailize particle filter class with inital particles
        self.filter = ParticleFilter(self.initial_particles)


    def set_initial_particles(self):
        initial_positions = np.zeros((self.num_particles, 3))
        rots = []

        for index, particle in enumerate(self.initial_particles_noise):
            # # # For Testing: Initialize at camera location
            # i = self.cam_states[0]
            # x = i[3]
            # y = i[7]
            # z = i[11]
            # rot1 = i.reshape(4,4)
            # rot = rot1[:3,:3]
            # gt_rotation_obj  = R.from_matrix(rot)
            # gt_euler  =  gt_rotation_obj.as_euler('xyz')
            # phi = gt_euler[0]
            # theta = gt_euler[1]
            # psi = gt_euler[2]



            # # For Testing: Initialize at camera location
            # i = self.cam_states[0]
            # x = i[3]+particle[0]
            # y = i[7]+particle[1]
            # z = i[11]+particle[2]
            # rot1 = i.reshape(4,4)
            # rot = rot1[:3,:3]
            # gt_rotation  = R.from_matrix(rot)
            # gt_euler  =  gt_rotation.as_euler('xyz')
            # phi = gt_euler[0]+ np.pi/4
            # theta = gt_euler[1]
            # psi = gt_euler[2]
            # if index < 10:
            #     print("ROTS1",phi,theta,psi)
            # gt_rotation_obj = R.from_euler('xyz',[phi,theta,psi])

            # For random particles within given bound
            x = self.initialization_center[0] + particle[0]
            y = self.initialization_center[1] + particle[1]
            z = self.initialization_center[2] + particle[2]
            phi   = self.initialization_center[3]+ particle[3]
            theta = self.initialization_center[4]+ particle[4]
            psi   = self.initialization_center[5]+ particle[5]
            gt_rotation_obj = R.from_euler('xyz',[phi,theta,psi])

            # # For random particles within given bound
            # x = particle[0]+self.initialization_center[0]
            # y = particle[1]+self.initialization_center[1]
            # z = particle[2]+self.initialization_center[2]
            # phi = particle[3]
            # theta = particle[4]
            # psi = particle[5]
            # gt_rotation_obj = R.from_euler('xyz',[phi,theta,psi])

            # set positions
            initial_positions[index,:] = [x,y,z]
            # set orientations, create rotation object
            rots.append(gt_rotation_obj)
    
        
        # # Create a 3D plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for index, particle in enumerate(self.initial_particles_noise):
        #     Rotating = rots[index].as_matrix()
        #     vector = np.dot(Rotating, np.array([1, 0, 0]))  # Unit vector along the x-axis
        #     ax.quiver(initial_positions[index][0], initial_positions[index][1], initial_positions[index][2], vector[0], vector[1], vector[2])

        # # Add camera initialization
        # initial_cam = self.cam_states[0].reshape(4,4)
        # vector = np.dot(initial_cam[:3,:3], np.array([1, 0, 0]))  # Unit vector along the x-axis
        # ax.quiver(initial_cam[0][3], initial_cam[1][3], initial_cam[2][3], vector[0], vector[1], vector[2], color='r')

        # # Set axis labels
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # # Set axis limits
        # ax.set_xlim([-2, 2])
        # ax.set_ylim([-2, 2])
        # ax.set_zlim([-2, 2])

        # # Add a legend
        # ax.legend()

        # # Show the 3D plot
        # plt.show()

        print("INITIAL POSITION ", initial_positions)
        print("INITIAL ROT", rots)
        return {'position':initial_positions, 'rotation':np.array(rots)}

    

    def move(self, x0=np.zeros(12), goal=np.zeros(12), dt=0.1):
        # integrate dynamics
        movement = self.control.simulate(x0, goal, dt)
        pass


    # def vector_visualization(self, )
    def odometry_update(self,state0, state1):
        state_difference = state1-state0
        rot0 = R.from_matrix(state0[:3,:3])
        rot1 = R.from_matrix(state1[:3,:3])
        eul0 = rot0.as_euler('xyz')
        eul1 = rot1.as_euler('xyz')
        diffeul = eul1-eul0

        odometry_particle_noise = np.random.uniform(np.array(
            [self.min_bounds['px'], self.min_bounds['py'], self.min_bounds['pz'], self.min_bounds['rz'], self.min_bounds['ry'], self.min_bounds['rx']]),
            np.array([self.max_bounds['px'], self.max_bounds['py'], self.max_bounds['pz'], self.max_bounds['rz'], self.max_bounds['ry'], self.max_bounds['rx']]),
            size = (self.num_particles, 6))
        

        for i in range(self.num_particles):
            odometry_particle_noise_translation = np.random.normal(0.0, 0.001,3)
            odometry_particle_noise_rotation = np.random.normal(0.0, 0.001,3)

            self.filter.particles['position'][i] += [state_difference[0][3]+odometry_particle_noise_translation[0], state_difference[1][3]+odometry_particle_noise_translation[0], state_difference[2][3]+odometry_particle_noise_translation[0]]

            peul = self.filter.particles['rotation'][i].as_euler('xyz')
            peul = peul + diffeul + odometry_particle_noise_rotation

            # Ensure the differences are within the range of -pi to pi
            peul[0] = (peul[0] + np.pi) % (2 * np.pi) - np.pi
            peul[1] = (peul[1] + np.pi) % (2 * np.pi) - np.pi
            peul[2] = (peul[2] + np.pi) % (2 * np.pi) - np.pi

            prot = R.from_euler('xyz',peul)
            self.filter.particles['rotation'][i] = prot
        
        print("Finish odometry update")

    def rgb_run(self,iter, img, current_state, msg=None, get_rays_fn=None, render_full_image=False):
        self.odometry_update(self.last_state,current_state)
        print("processing image")
        start_time = time.time()

        # make copies to prevent mutations
        particles_position_before_update = np.copy(self.filter.particles['position'])
        particles_rotation_before_update = np.copy(self.filter.particles['rotation'])

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
            loss_pose = np.zeros((4,4))
            rot = particles_rotation_before_update[index].as_matrix()
            loss_pose[0:3, 0:3] = rot
            loss_pose[0:3,3] = particle[0:3]
            loss_pose[3,3] = 1.0
            loss_poses.append(loss_pose)

        losses, nerf_time = self.nerf.get_loss(loss_poses, batch, img, iter=iter)
        print("Pass losses")
        # print("Loss Values" ,losses)
        temp = 0
        for index, particle in enumerate(particles_position_before_update):
            self.filter.weights[index] = 1/(losses[index]+temp)

        total_nerf_time += nerf_time

        # Resample Weights
        
        position_est = self.filter.compute_simple_position_average()
        quat_est = self.filter.compute_simple_rotation_average()
        pose_est = np.zeros(3+4)  # Initialize as identity matrix
        pose_est[:3] = position_est 
        pose_est[3:] = quat_est 
        self.all_pose_est.append(pose_est)

        self.filter.update()
        self.num_updates += 1

        # # Create a 3D plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for index in range(self.num_particles):
        #     Rotating = self.filter.particles['rotation'][index].as_matrix()
        #     vector = np.dot(Rotating, np.array([1, 0, 0]))  # Unit vector along the x-axis
        #     ax.quiver(self.filter.particles['position'][index][0], self.filter.particles['position'][index][1], self.filter.particles['position'][index][2], vector[0], vector[1], vector[2])

        # # Add camera 
        # initial_cam = self.cam_states[iter].reshape(4,4)
        # vector = np.dot(initial_cam[:3,:3], np.array([1, 0, 0]))  # Unit vector along the x-axis
        # ax.quiver(initial_cam[0][3], initial_cam[1][3], initial_cam[2][3], vector[0], vector[1], vector[2], color='r')

        # # Pose est 
        # estimated_rot = R.from_quat(pose_est[3:])
        # vector = np.dot(estimated_rot.as_matrix(), np.array([1, 0, 0]))  # Unit vector along the x-axis
        # ax.quiver(pose_est[0], pose_est[1], pose_est[2], vector[0], vector[1], vector[2], color='g')

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # # ax.set_xlim([-2, 2])
        # # ax.set_ylim([-2, 2])
        # # ax.set_zlim([-2, 2])
        # ax.legend()
        # plt.show()

        # Update odometry step
        # current_state = self.cam_states[iter].reshape(4,4)
        # next_state = self.cam_states[iter+1].reshape(4,4)
        # self.publish_pose_est(pose_est)


        update_time = time.time() - start_time
        print("forward passes took:", total_nerf_time, "out of total", update_time, "for update step")

        return pose_est

    def step(self, state, flag = False):
        cam2world = np.zeros((3,4))
        cam2world[:,3] = state[:3]
        if not flag:
            rot_mat = R.from_euler('xyz',[state[3]+np.pi/2, state[4], state[5]-np.pi/2]).as_matrix()
        else:
            rot_mat = R.from_euler('xyz',[state[3], state[4], state[5]]).as_matrix()
        cam2world[:3,:3] = rot_mat

        base_img = self.nerf.render_Nerf_image_base(cam2world,save=False, save_name = "base", iter=iter, particle_number=None)
        # cv2.imshow("img ",base_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        tmp = np.vstack((cam2world, np.array([[0,0,0,1]])))
        if self.last_state is None:
            self.last_state = copy.deepcopy(tmp)
        pose_est = self.rgb_run(iter, base_img, tmp) 
        self.last_state = copy.deepcopy(tmp)
        pos = pose_est[:3]
        rpy = R.from_quat(pose_est[3:]).as_euler('xyz')
        if not flag:
            res = np.array([pos[0], pos[1], pos[2], rpy[0]-np.pi/2, rpy[1], rpy[2]+np.pi/2])
        else:
            res = np.array([pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]])
            
        return res
 
if __name__ == "__main__":

    camera_path = './NeRF_UAV_simulation/camera_path.json'
    nerf_file_path = './outputs/IRL2/nerfacto/2023-09-21_210511/config.yml'

    mcl = Run(camera_path,nerf_file_path, 80, 80, 50)      

    # mcl.mat3d()
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
        cam_states = np.array(mcl.cam_states[iter]).reshape((4,4))
        rpy = R.from_matrix(cam_states[0:3, 0:3]).as_euler('xyz')
        state = np.concatenate((cam_states[:,3], rpy))
        # base_img = mcl.nerf.render_Nerf_image_simple(mcl.cam_states[iter],mcl.cam_states[iter+1],save=False, save_name = "base", iter=iter, particle_number=None)
        # # cv2.imshow("img ",base_img)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()

    
        # pose_est = mcl.rgb_run(iter, base_img)   
        pose_est = mcl.step(state, flag = True)

        ########################## Error Visualization ##########################
        est_states[iter] = pose_est[0:3]
        gt_states[iter] = mcl.cam_states[iter]

        est_rotation_obj = R.from_euler('xyz', pose_est[3:])
        est_euler[iter] = est_rotation_obj.as_euler('xyz', degrees=True)

        gt_matrix = gt_states[iter].reshape(4,4)
        gt_rotation_obj  = mcl.nerf.base_rotations[iter]
        gt_euler[iter]  =  gt_rotation_obj.as_euler('xyz', degrees=True)
           
    
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
        
        print(f'cam_states iteration {iter}:\n',mcl.cam_states[iter])
        print(f'pose est iteration {iter}:\n',pose_est)

    print("########################Done########################")
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt_states[:iter+1,3], gt_states[:iter+1,7], gt_states[:iter+1,11], 'g')
    ax.scatter(gt_states[:iter+1,3], gt_states[:iter+1,7], gt_states[:iter+1,11], 'm')
    ax.plot(est_states[:iter+1,0], est_states[:iter+1,1], est_states[:iter+1,2], 'r')
    ax.scatter(est_states[:iter+1,0], est_states[:iter+1,1], est_states[:iter+1,2], 'm')
    plt.show()
