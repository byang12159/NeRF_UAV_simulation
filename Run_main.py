import numpy as np
import cv2
import time
import gtsam
import matplotlib.pyplot as plt
import time
import json

from particle_filter import ParticleFilter
from utils import get_pose
from full_filter import NeRF

from controller import Controller

# update step:

# particlefilter.py update()

# this is called in nav_mode.py in rgb_run()

class Run():
    def __init__(self, camera_path):

        # Import camera path json file with information about camera
        with open(camera_path, 'r') as file:
            data = json.load(file)

        cam_data = data.get('camera_path')
        self.cam_states = np.zeros((len(cam_data),16))

        for j in range(len(cam_data)):
            self.cam_states[j] = cam_data[0].get('camera_to_world')

        self.nerfW = data.get('render_height')
        self.nerfH = data.get('render_width')
        self.nerfFov = (data.get('keyframes')[0].get('fov')) #Assuming all cameras rendered using same FOV 

        print("Finish importing camera states")


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

        self.control = Controller()
        self.nerf = NeRF()

        self.view_debug_image_iteration = 0 #view NeRF rendered image at estimated pose after number of iterations (set to 0 to disable)

        self.get_initial_distribution()

        # add initial pose estimate before 1st update step
        position_est = self.filter.compute_weighted_position_average()
        rot_est = self.filter.compute_simple_rotation_average()
        # TODO: issue with Pose3
        # pose_est = gtsam.Pose3(rot_est, position_est).matrix()
        # self.all_pose_est.append(pose_est)
        
    def get_initial_distribution(self):
        # NOTE for now assuming everything stays in NeRF coordinates (x right, y up, z inward)

        # get distribution of particles from user
        self.initial_particles_noise = np.random.uniform(np.array(
            [self.min_bounds['px'], self.min_bounds['py'], self.min_bounds['pz'], self.min_bounds['rz'], self.min_bounds['ry'], self.min_bounds['rx']]),
            np.array([self.max_bounds['px'], self.max_bounds['py'], self.max_bounds['pz'], self.max_bounds['rz'], self.max_bounds['ry'], self.max_bounds['rx']]),
            size = (self.num_particles, 6))

        self.initial_particles = self.set_initial_particles()
        self.filter = ParticleFilter(self.initial_particles)
        # print(self.initial_particles)

    def set_initial_particles(self):
        initial_positions = np.zeros((self.num_particles, 3))
        rots = []
        for index, particle in enumerate(self.initial_particles_noise):
            x = particle[0]
            y = particle[1]
            z = particle[2]
            phi = particle[3]
            theta = particle[4]
            psi = particle[5]

            # print(x,y,z)
            particle_pose = get_pose(phi, theta, psi, x, y, z, self.obs_img_pose, self.center_about_true_pose)
            
            # set positions
            initial_positions[index,:] = [particle_pose[0,3], particle_pose[1,3], particle_pose[2,3]]
            # set orientations
            rots.append(gtsam.Rot3(particle_pose[0:3,0:3]))
            # print(initial_particles)

        return {'position':initial_positions, 'rotation':np.array(rots)}

    def move(self, x0=np.zeros(12), goal=np.zeros(12), dt=0.1):
        # integrate dynamics
        movement = self.control.simulate(x0, goal, dt)
        pass

    def rgb_run(self, img,msg=None, get_rays_fn=None, render_full_image=False):
        print("processing image")
        start_time = time.time()
        self.rgb_input_count += 1

        # make copies to prevent mutations
        particles_position_before_update = np.copy(self.filter.particles['position'])
        particles_rotation_before_update = [gtsam.Rot3(i.matrix()) for i in self.filter.particles['rotation']]

        # if self.use_convergence_protection:
        #     for i in range(self.number_convergence_particles):
        #         t_x = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
        #         t_y = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
        #         t_z = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
        #         # TODO this is not thread safe. have two lines because we need to both update
        #         # particles to check the loss and the actual locations of the particles
        #         self.filter.particles["position"][i] = self.filter.particles["position"][i] + np.array([t_x, t_y, t_z])
        #         particles_position_before_update[i] = particles_position_before_update[i] + np.array([t_x, t_y, t_z])

        
        # resize input image so it matches the scale that NeRF expects
        img = cv2.resize(img, (int(self.nerfW), int(self.nerfH)))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.nerf.obs_img = img
        show_true = self.view_debug_image_iteration != 0 and self.num_updates == self.view_debug_image_iteration-1
        #Create a grid to sample image points for comparison
        self.nerf.get_poi_interest_regions(show_true, self.sampling_strategy)
        # plt.imshow(self.nerf.obs_img)
        # plt.show()

        total_nerf_time = 0

        # if self.sampling_strategy == 'random':
        rand_inds = np.random.choice(self.nerf.coords.shape[0], size=self.nerf.batch_size, replace=False)
        batch = self.nerf.coords[rand_inds]

        loss_poses = []
        for index, particle in enumerate(particles_position_before_update):
            loss_pose = np.zeros((4,4))
            rot = particles_rotation_before_update[index]
            loss_pose[0:3, 0:3] = rot.matrix()
            loss_pose[0:3,3] = particle[0:3]
            loss_pose[3,3] = 1.0
            loss_poses.append(loss_pose)
        losses, nerf_time = self.nerf.get_loss(loss_poses, batch)
   
        for index, particle in enumerate(particles_position_before_update):
            self.filter.weights[index] = 1/losses[index]
        total_nerf_time += nerf_time

        # Resample Weights
        self.filter.update()
        self.num_updates += 1
        print("UPDATE STEP NUMBER", self.num_updates, "RAN")
        print("number particles:", self.num_particles)

        avg_pose = self.filter.compute_weighted_position_average()
        avg_rot = self.filter.compute_simple_rotation_average()
        self.nerf_pose = gtsam.Pose3(avg_rot, gtsam.Point3(avg_pose[0], avg_pose[1], avg_pose[2])).matrix()

        if self.plot_particles:
            self.visualize()
        
        position_est = self.filter.compute_weighted_position_average()
        rot_est = self.filter.compute_simple_rotation_average()
        pose_est = gtsam.Pose3(rot_est, position_est).matrix()

        if self.log_results:
            self.all_pose_est.append(pose_est)
        
        if not self.run_inerf_compare:
            img_timestamp = msg.header.stamp
            self.publish_pose_est(pose_est, img_timestamp)
        else:
            self.publish_pose_est(pose_est)
    
        update_time = time.time() - start_time
        print("forward passes took:", total_nerf_time, "out of total", update_time, "for update step")

        if not self.run_predicts:
            self.filter.predict_no_motion(self.px_noise, self.py_noise, self.pz_noise, self.rot_x_noise, self.rot_y_noise, self.rot_z_noise) #  used if you want to localize a static image
        
        

        # return is just for logging
        return pose_est



if __name__ == "__main__":

    camera_path = 'camera_path-2.json'

    mcl = Run(camera_path)      
    # Initializes particles in get_initial_distribution()

    # Initialize Drone Position
    drone_state = [[0,0,0,0,0,0]]

    # Assume constant time step between trajectory stepping
    for iter in range(len(mcl.cam_states)):
        # MCL: Prediction Step
        mcl.move()

        # MCL: Update and Resample Steps
        mcl.rgb_run(mcl.cam_states[iter])

    print("########################Done########################")