import numpy as np 
from numpy import cos, sin
from verse.agents import BaseAgent
from verse.parser.parser import ControllerIR
import copy 
import matplotlib.pyplot as plt 
import scipy
from scipy.integrate import odeint
import os 
import json 
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation as R
from nerfstudio.cameras.cameras import Cameras, CameraType
import torch 
from nerfstudio.utils import colormaps
import cv2
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
script_dir = os.path.dirname(os.path.realpath(__file__))
from Run_main import Run
from camera_path_spline import spline

import logging
from datetime import datetime

import pickle 

def render_Nerf_image_simple(model, camera_to_world, save, save_name, iter,particle_number):
    # print("SIMPLE RENDER C2W ...........\n",camera_to_world)
    fov = 50

    width = height = 320

    fx = (width/2)/(np.tan(np.deg2rad(fov)/2))
    fy = (height/2)/(np.tan(np.deg2rad(fov)/2))
    cx = width/2
    cy = height/2
    nerfW = nerfH = width

    camera_to_world = torch.FloatTensor( camera_to_world )

    camera = Cameras(camera_to_worlds = camera_to_world, fx = fx, fy = fy, cx = cx, cy = cy, width=nerfW, height=nerfH, camera_type=CameraType.PERSPECTIVE)
    camera = camera.to('cuda')
    ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)

    with torch.no_grad():
        tmp = model.get_outputs_for_camera_ray_bundle(ray_bundle)

    img = tmp['rgb']
    img =(colormaps.apply_colormap(image=img, colormap_options=colormaps.ColormapOptions())).cpu().numpy()
    # plt.imshow(img)
    # plt.show()

    if save:
        output_dir = os.path.join(script_dir, f"./imgs/{save_name}{particle_number}.jpg")
        cv2.imwrite(output_dir, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)*255)
    return img


# quadrotor physical constants
g = 9.81; d0 = 10; d1 = 8; n0 = 10; kT = 0.91

# non-linear dynamics
def dynamics(state, u):
    x, vx, theta_x, omega_x, y, vy, theta_y, omega_y, z, vz, theta_z, omega_z = state.reshape(-1).tolist()
    ax, ay, F, az = u.reshape(-1).tolist()
    dot_x = np.array([
     cos(theta_z)*vx-sin(theta_z)*vy,
     g * np.tan(theta_x),
     -d1 * theta_x + omega_x,
     -d0 * theta_x + n0 * ax,
     sin(theta_z)*vx+cos(theta_z)*vy,
     g * np.tan(theta_y),
     -d1 * theta_y + omega_y,
     -d0 * theta_y + n0 * ay,
     vz,
     kT * F - g,
     omega_z,
     n0 * az])
    return dot_x

# linearization

# The state variables are x, y, z, vx, vy, vz, theta_x, theta_y, omega_x, omega_y
A = np.zeros([10, 10])
A[0, 1] = 1.
A[1, 2] = g
A[2, 2] = -d1
A[2, 3] = 1
A[3, 2] = -d0
A[4, 5] = 1.
A[5, 6] = g
A[6, 6] = -d1
A[6, 7] = 1
A[7, 6] = -d0
A[8, 9] = 1.

B = np.zeros([10, 3])
B[3, 0] = n0
B[7, 1] = n0
B[9, 2] = kT

class DroneAgent(BaseAgent):
    def __init__(self, id, code=None, file_name = None, ref_spline = 'camera_path_spline'):
        self.id = id
        self.init_cont = None
        self.init_disc = None
        self.static_parameters = None 
        self.uncertain_parameters = None
        self.decision_logic = ControllerIR.empty()

        # with open(ref_spline, 'r') as f:
        #     data = json.load(f)
        data = spline
        tks_x = (data['x']['0'],data['x']['1'],data['x']['2'])
        tks_y = (data['y']['0'],data['y']['1'],data['y']['2'])
        tks_z = (data['z']['0'],data['z']['1'],data['z']['2'])

        spline_x = UnivariateSpline._from_tck(tks_x)
        spline_y = UnivariateSpline._from_tck(tks_y)
        spline_z = UnivariateSpline._from_tck(tks_z)

        self.ref_traj = [spline_x, spline_y, spline_z]

        def lqr(A, B, Q, R):
            """Solve the continuous time lqr controller.
            dx/dt = A x + B u
            cost = integral x.T*Q*x + u.T*R*u
            """
            # http://www.mwm.im/lqr-controllers-with-python/
            # ref Bertsekas, p.151

            # first, try to solve the ricatti equation
            X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

            # compute the LQR gain
            K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

            eigVals, eigVecs = scipy.linalg.eig(A - B * K)

            return np.asarray(K), np.asarray(X), np.asarray(eigVals)
        
        ####################### solve LQR #######################
        n = A.shape[0]
        m = B.shape[1]
        Q = np.eye(n)
        Q[0, 0] = 100.
        Q[4, 4] = 100.
        Q[8, 8] = 100.
        # Q[11,11] = 0.01
        R = np.diag([1., 1., 1.])
        self.K, _, _ = lqr(A, B, Q, R)

    def u(self, x, goal):
        yaw = x[10]
        err = [goal[0],0,0,0, goal[1],0,0,0, goal[2],0] - x[:10]
        err_pos = err[[0,4,8]]

        err_pos = np.linalg.inv(np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0,0,1]
        ]))@err_pos

        err[[0,4,8]] = err_pos
        u_pos = self.K.dot(err) + [0, 0, g / kT]
        u_ori = (goal[3]-yaw)*2+(0-x[11])*1.0
        # if abs(goal[3]-yaw)>1:
        #     print('stop')
        return np.concatenate((u_pos, [u_ori]))
    
    ######################## The closed_loop system #######################
    def cl_nonlinear(self, x, t, x_est, goal):
        x = np.array(x)
        dot_x = dynamics(x, self.u(x_est, goal))
        return dot_x

    # simulate
    def simulate(self, x, x_est, goal, dt):
        curr_position = np.array(x)[[0, 4, 8]]
        goal_pos = goal[:3]
        error = goal_pos - curr_position
        distance = np.sqrt((error**2).sum())
        if distance > 1:
            goal[:3] = curr_position + error / distance
        return odeint(self.cl_nonlinear, x, [0, dt], args=(x_est, goal,))[-1]
    
    def step(self, cur_state_estimated, initial_condition, time_step, goal_state):
        goal_pos = [
            self.ref_traj[0](goal_state[0]),
            self.ref_traj[1](goal_state[0]),
            self.ref_traj[2](goal_state[0]),
        ]
        goal_yaw = np.arctan2(
            self.ref_traj[1](goal_state[0]+0.001)-self.ref_traj[1](goal_state[0]),
            self.ref_traj[0](goal_state[0]+0.001)-self.ref_traj[0](goal_state[0])
        ) 
        goal_yaw = goal_yaw%(np.pi*2)
        if goal_yaw > np.pi/2:
            goal_yaw -= 2*np.pi
        goal = goal_pos + [goal_yaw]
        sol = self.simulate(initial_condition, cur_state_estimated, goal, time_step)
        self.last_ref_yaw = goal_yaw
        return sol

    def run_ref(self, ref_state, time_step):
        ref_pos = ref_state[0]
        ref_v = ref_state[1]
        return np.array([ref_pos+ref_v*time_step, ref_v])

    def TC_simulate(self, mode, initial_condition, time_horizon, time_step, lane_map = None):
        time_steps = np.arange(0, time_horizon+time_step/2, time_step)

        state = np.array(initial_condition)
        trajectory = copy.deepcopy(state)
        trajectory = np.insert(trajectory, 0, time_steps[0])
        trajectory = np.reshape(trajectory, (1, -1))
        for i in range(1, len(time_steps)):
            x_ground_truth = state[:12]
            x_estimate = state[12:24]
            ref_state = state[24:]
            x_next = self.step(x_estimate, x_ground_truth, time_step, ref_state)
            x_next[10] = x_next[10]%(np.pi*2)
            if x_next[10] > np.pi/2:
                x_next[10] = x_next[10]-np.pi*2
            ref_next = self.run_ref(ref_state, time_step)
            state = np.concatenate((x_next, x_estimate, ref_next))
            tmp = np.insert(state, 0, time_steps[i])
            tmp = np.reshape(tmp, (1,-1))
            trajectory = np.vstack((trajectory, tmp))

        return trajectory
def sample_environmental_parameters(E):
    Ep = np.random.choice(len(E))
    e = np.random.uniform(E[Ep, 0,:], E[Ep,1,:])
    return e

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(script_dir, './verse/exp2_safe.pickle'), 'rb') as f:
        M, E, _, _ = pickle.load(f)

    E = np.array([
        [
            [0, -0.1],
            [0.1, 0]
        ]
    ])
    fn = os.path.join(script_dir, './camera_path.json')
    with open(fn, 'r') as f:
        data = json.load(f)
    cam_data = data.get('camera_path')

    cam_states = np.zeros((len(cam_data),16))

    for j in range(len(cam_data)):
        cam_states[j] = np.array(cam_data[j].get('camera_to_world'))

    spline_fn = os.path.join(script_dir, './camera_path_spline.json')    
    drone_agent = DroneAgent('drone',ref_spline=spline_fn)
    
    # for change_env_fog in range(0,10,1):
    #     for change_env_dark in range(0,10,1):
    traj_list = []
    estimate_traj_list = []
    init_list = []
    for i in range(20):
        e = sample_environmental_parameters(E)
        param_fog = e[0]
        param_dark = e[1]

        #Run each path twice with slighly perturbed env variables
        
        cam_init = cam_states[0].reshape(4,4)
        cam_init_pos = cam_init[0:3, 3]
        cam_rpy = R.from_matrix(cam_init[0:3, 0:3]).as_euler('xyz')

        drone_init = np.array([
            cam_init_pos[0], 0, cam_rpy[0]-np.pi/2, 0, 
            cam_init_pos[1], 0, cam_rpy[1], 0, 
            cam_init_pos[2], 0, cam_rpy[2]+np.pi/2, 0, 
        ])
        tmp = np.array([
            0.01, 0, 0.01, 0,
            0.01, 0, 0.01, 0,
            0.01, 0, 0.01, 0,
        ])
        X_0 = np.vstack((drone_init.reshape((1,-1))-tmp, drone_init.reshape((1,-1))+tmp))

        init_point = np.random.uniform(X_0[0,:], X_0[1,:])

        ref_init = np.array([0,1])

        state = init_point 
        ref = ref_init 
        traj = np.concatenate(([0], init_point, init_point, ref_init)).reshape((1,-1))

        init_list.append((init_point, e))
        
        camera_path = './NeRF_UAV_simulation/camera_path.json'
        config_fn = './outputs/IRL2/nerfacto/2023-09-21_210511/config.yml'
        mcl = Run(camera_path,config_fn, 80, 80, 50)      
        est = []
        gt = []
        for j in range(300):
            print(f"======> Iter {i} {j}")
            gt_state = np.array([state[0], state[4], state[8], state[2], state[6], state[10]])
            est_state = mcl.step(gt_state, 0, param_fog, param_dark)
            gt.append(gt_state)
            est.append(est_state)
            est_state[5] = est_state[5]%(2*np.pi)
            if est_state[5] > np.pi/2:
                est_state[5] -= np.pi*2
            est_state_full = np.array([
                est_state[0],
                state[1],
                # est_state[3],
                state[2],
                state[3],
                est_state[1],
                state[5],
                # est_state[4],
                state[6],
                state[7],
                est_state[2],
                state[9],
                est_state[5],
                state[11],
            ])
            init = np.concatenate((state, est_state_full, ref))
            trace = drone_agent.TC_simulate(None, init, 0.1, 0.01)
            state = trace[-1,1:13]
            ref = trace[-1, 25:]
            lstate = trace[-1,1:]
            ltime = i*0.1
            lstate = np.insert(lstate, 0, ltime).reshape((1,-1))
            traj = np.vstack((traj,lstate))

        traj_list.append(gt)
        estimate_traj_list.append(est)
        with open(os.path.join(script_dir, './verse/vcs_sim_exp1_test.pickle'),'wb+') as f:
            pickle.dump(traj_list, f)
        with open(os.path.join(script_dir, './verse/vcs_estimate_exp1_test.pickle'),'wb+') as f:
            pickle.dump(estimate_traj_list, f)
        with open(os.path.join(script_dir, './verse/vcs_init_exp1_test.pickle'), 'wb+') as f:
            pickle.dump(init_list, f)