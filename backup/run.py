import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import copy
import argparse

from controller import simulate, angle  
import dynamics3 

parser = argparse.ArgumentParser(description="")
parser.add_argument('--initerror', type=float, default=0., help='error in initial position.')
parser.add_argument('--save', type=str, help='filename to save the data.')
args = parser.parse_args()

def get_initial_distribution():
    # NOTE for now assuming everything stays in NeRF coordinates (x right, y up, z inward)
    if self.run_inerf_compare:
        # for non-global loc mode, get random pose based on iNeRF evaluation method from their paper
        # sample random axis from unit sphere and then rotate by a random amount between [-40, 40] degrees
        # translate along each axis by a random amount between [-10, 10] cm
        rot_rand = 40.0
        if self.global_loc_mode:
            trans_rand = 1.0
        else:
            trans_rand = 0.1
        
        # get random axis and angle for rotation
        x = np.random.rand()
        y = np.random.rand()
        z = np.random.rand()
        axis = np.array([x,y,z])
        axis = axis / np.linalg.norm(axis)
        angle = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
        euler = (gtsam.Rot3.AxisAngle(axis, angle)).ypr()

        # get random translation offset
        t_x = np.random.uniform(low=-trans_rand, high=trans_rand)
        t_y = np.random.uniform(low=-trans_rand, high=trans_rand)
        t_z = np.random.uniform(low=-trans_rand, high=trans_rand)

        # use initial random pose from previously saved log
        if self.use_logged_start:
            log_file = self.log_directory + "/" + "initial_pose_" + self.model_name + "_" + str(self.obs_img_num) + "_" + "poses.npy"
            start = np.load(log_file)
            print(start)
            euler[0], euler[1], euler[2], t_x, t_y, t_z = start

        # log initial random pose
        elif self.log_results:
            with open(self.log_directory + "/" + "initial_pose_" + self.model_name + "_" + str(self.obs_img_num) + "_" + "poses.npy", 'wb') as f:
                np.save(f, np.array([euler[0], euler[1], euler[2], t_x, t_y, t_z]))

        if self.global_loc_mode:
            # 360 degree rotation distribution about yaw
            self.initial_particles_noise = np.random.uniform(np.array([-trans_rand, -trans_rand, -trans_rand, 0, -179, 0]), np.array([trans_rand, trans_rand, trans_rand, 0, 179, 0]), size = (self.num_particles, 6))
        else:
            self.initial_particles_noise = np.random.uniform(np.array([-trans_rand, -trans_rand, -trans_rand, 0, 0, 0]), np.array([trans_rand, trans_rand, trans_rand, 0, 0, 0]), size = (self.num_particles, 6))

        # center translation at randomly sampled position
        self.initial_particles_noise[:, 0] += t_x
        self.initial_particles_noise[:, 1] += t_y
        self.initial_particles_noise[:, 2] += t_z

        if not self.global_loc_mode:
            for i in range(self.initial_particles_noise.shape[0]):
                # rotate random 3 DOF rotation about initial random rotation for each particle
                n1 = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
                n2 = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
                n3 = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
                euler_particle = gtsam.Rot3.AxisAngle(axis, angle).retract(np.array([n1, n2, n3])).ypr()

                # add rotation noise for initial particle distribution
                self.initial_particles_noise[i,3] = euler_particle[0] * 180.0 / np.pi
                self.initial_particles_noise[i,4] = euler_particle[1] * 180.0 / np.pi 
                self.initial_particles_noise[i,5] = euler_particle[2] * 180.0 / np.pi  
    
    else:
        # get distribution of particles from user
        self.initial_particles_noise = np.random.uniform(np.array(
            [self.min_bounds['px'], self.min_bounds['py'], self.min_bounds['pz'], self.min_bounds['rz'], self.min_bounds['ry'], self.min_bounds['rx']]),
            np.array([self.max_bounds['px'], self.max_bounds['py'], self.max_bounds['pz'], self.max_bounds['rz'], self.max_bounds['ry'], self.max_bounds['rx']]),
            size = (self.num_particles, 6))

    self.initial_particles = self.set_initial_particles()
    self.filter = ParticleFilter(self.initial_particles)

def set_initial_particles(num_particles):
    initial_positions = np.zeros((num_particles, 3))
  
    rots = []
    for index, particle in enumerate(self.initial_particles_noise):
        x = particle[0]
        y = particle[1]
        z = particle[2]
        phi = particle[3]
        theta = particle[4]
        psi = particle[5]

        particle_pose = get_pose(phi, theta, psi, x, y, z, self.nerf.obs_img_pose, self.center_about_true_pose)
        
        # set positions
        initial_positions[index,:] = [particle_pose[0,3], particle_pose[1,3], particle_pose[2,3]]
        # set orientations
        rots.append(gtsam.Rot3(particle_pose[0:3,0:3]))
        # print(initial_particles)
    return {'position':initial_positions, 'rotation':np.array(rots)}

class Core:
    def __init__(self, num_particles):
        self.initial_particles = set_initial_particles(num_particles)
        # self.filter = ParticleFilter(self.initial_particles)

def main():
    pass

if __name__ == "__main__":
    core = Core()

