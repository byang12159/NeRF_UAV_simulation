import numpy as np
from scipy.linalg import logm,expm
from scipy.spatial.transform import Rotation as R
from multiprocessing import Lock

class ParticleFilter:

    def __init__(self, initial_particles):
        self.num_particles=len(initial_particles['position'])
        self.particles = initial_particles
        self.weights=np.ones(self.num_particles)
        self.particle_lock = Lock()

    def reduce_num_particles(self, num_particles):
        self.particle_lock.acquire()
        self.num_particles = num_particles
        self.weights = self.weights[0:num_particles]
        self.particles['position'] = self.particles['position'][0:num_particles]
        self.particles['rotation'] = self.particles['rotation'][0:num_particles]
        self.particle_lock.release()

    def predict_no_motion(self, p_x, p_y, p_z, r_x, r_y, r_z):
        self.particle_lock.acquire()
        self.particles['position'][:,0] += p_x * np.random.normal(size = (self.particles['position'].shape[0]))
        self.particles['position'][:,1] += p_y * np.random.normal(size = (self.particles['position'].shape[0]))
        self.particles['position'][:,2] += p_z * np.random.normal(size = (self.particles['position'].shape[0]))

        # TODO see if this can be made faster
        for i in range(len(self.particles['rotation'])):
            n1 = r_x * np.random.normal()
            n2 = r_y * np.random.normal()
            n3 = r_z * np.random.normal()
            self.particles['rotation'][i] = self.particles['rotation'][i].retract(np.array([n1, n2, n3]))
        self.particle_lock.release()

    def predict_with_delta_pose(self, delta_pose, p_x, p_y, p_z, r_x, r_y, r_z):

        # TODO see if this can be made faster
        delta_rot_t_tp1= delta_pose.rotation()
        for i in range(len(self.particles['rotation'])):
            # TODO do rotation in gtsam without casting to matrix
            pose = np.eye(4)  
            pose[:3, :3] = self.particles['rotation'][i] 
            pose[:3, 3] = self.particles['position'][i]

            new_pose = pose @ delta_pose
            new_position = new_pose.translation()
            self.particles['position'][i][0] = new_position[0]
            self.particles['position'][i][1] = new_position[1]
            self.particles['position'][i][2] = new_position[2]
            self.particles['rotation'][i] = new_pose.rotation()

            n1 = r_x * np.random.normal()
            n2 = r_y * np.random.normal()
            n3 = r_z * np.random.normal()
    
            # self.particles['rotation'][i] = gtsam.Rot3(self.particles['rotation'][i].retract(np.array([n1, n2, n3])).matrix())

        self.particles['position'][:,0] += (p_x * np.random.normal(size = (self.particles['position'].shape[0])))
        self.particles['position'][:,1] += (p_y * np.random.normal(size = (self.particles['position'].shape[0])))
        self.particles['position'][:,2] += (p_z * np.random.normal(size = (self.particles['position'].shape[0])))


    def update(self):
        # use fourth power
        self.weights = np.square(self.weights)
        self.weights = np.square(self.weights)

        # normalize weights
        sum_weights=np.sum(self.weights)
        # print("pre-normalized weight sum", sum_weights)
        self.weights=self.weights / sum_weights
    
        #resample

        out = int(0.5*self.num_particles) # Number of outlier particles chosen
        # choice = np.random.choice(self.num_particles, self.num_particles-out, p = self.weights, replace=True)
        choice = np.random.choice(self.num_particles, self.num_particles-out, p = self.weights, replace=True)
        temp = {'position':np.copy(self.particles['position'])[choice, :], 'rotation':np.copy(self.particles['rotation'])[choice]}

        pos_est = self.compute_simple_position_average()
        rot_est = self.compute_simple_rotation_average()
        # Add some particles spread 
        for i in range(out):
            resample_particle_noise_translation = np.random.uniform(-0.05, 0.05,3)

            resample_particle_noise_rotation = np.random.normal(-0.05, 0.05,3)
            temp['position'] = np.concatenate((temp['position'],(pos_est + resample_particle_noise_translation).reshape((1,-1))),axis=0)
            temp['rotation'] = np.append(temp['rotation'],R.from_euler('xyz', R.from_quat(rot_est).as_euler('xyz')+resample_particle_noise_rotation))

        self.particles = temp


    def compute_simple_position_average(self):
        # Simple averaging does not use weighted average or k means.
        avg_pose = np.average(self.particles['position'], axis=0)
        return avg_pose

    def compute_weighted_position_average(self):
        print("weights",self.weights)
        avg_pose = np.average(self.particles['position'], weights=self.weights, axis=0)
        return avg_pose
    
    def compute_simple_rotation_average(self):
        # Simple averaging does not use weighted average or k means.
        # https://users.cecs.anu.edu.au/~hartley/Papers/PDF/Hartley-Trumpf:Rotation-averaging:IJCV.pdf section 5.3 Algorithm 1
        rotations = self.particles['rotation']

        quaternion = np.zeros((self.num_particles,4))
        for i in range(self.num_particles):
            quaternion[i] = rotations[i].as_quat()

        weights = np.ones(self.num_particles)
        #Using multiple quaternion average technique https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions
        avg_quat = np.linalg.eigh(np.einsum('ij,ik,i->...jk', quaternion, quaternion, weights))[1][:, -1]
        return avg_quat

        # total_quat = np.zeros(4)
        # for i in range(self.num_particles):
        #     total_quat += rotations[i].as_quat()
        
        # avg_quat = total_quat/self.num_particles

        # # Ensure the differences are within the range of -pi to pi
        # yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        # pitch_diff = (pitch_diff + np.pi) % (2 * np.pi) - np.pi
        # roll_diff = (roll_diff + np.pi) % (2 * np.pi) - np.pi

        # return [yaw_diff, pitch_diff, roll_diff]
        # rotobj = R.from_quat(avg_quat)
        
        # return rotation object
        # return rotobj

        # epsilon = 0.00001
        # max_iters = 10
        # rotations = self.particles['rotation']

        # R = rotations[0].as_matrix()
        # for i in range(max_iters):
        #     rot_sum = np.zeros((3))
        #     for rot in rotations:
        #         rot_sum = rot_sum  + logm(R.T @ rot.as_matrix())

        #     r = rot_sum / len(rotations)
        #     if np.linalg.norm(r) < epsilon:
        #         # print("rotation averaging converged at iteration: ", i)
        #         # print("average rotation: ", R)
        #         return R
        #     else:
        #         # TODO do the matrix math in gtsam to avoid all the type casting
        #         R = R @ expm(r)

    # def odometry_update(self,state_difference):
    #     for i in range(self.num_particles):
    #         self.particles['position'][i] += [state_difference[3], state_difference[7], state_difference[11]]

    #         diff_rotation = R.from_matrix(self.cam_states[i].reshape(4,4)[0:3,0:3])
    #         self.particles['rotation'][i] *= diff_rotation
        
    #     print("Finish odometry update")