from verse.plotter.plotter2D import * 
from verse.plotter.plotter3D import *

from drone_agent import DroneAgent
from drone_agent_pc import DroneAgentPC
from verse import Scenario, ScenarioConfig
from enum import Enum, auto 

import copy 
import os 
import torch
import numpy as np 
from typing import Tuple
import matplotlib.pyplot as plt 
import polytope as pc
import itertools
import scipy.spatial
from datetime import datetime 
from verse.analysis.verifier import ReachabilityMethod

from get_all_models import get_all_models

import pickle 
import json 
import ray
import plotly.graph_objects as go 
import pyvista as pv 

from scipy.spatial.transform import Rotation as R

script_dir = os.path.realpath(os.path.dirname(__file__))

class DroneMode(Enum):
    Normal = auto() 

class State:
    x: float 
    vx: float 
    roll: float 
    vroll: float 
    y: float 
    vy: float 
    pitch: float 
    vpitch: float 
    z: float 
    vz: float 
    yaw: float 
    vyaw: float 
    pos_ref: float 
    v_ref: float 
    mode: DroneMode 

    def __init__(self, 
        x, vx, roll, vroll, 
        y, vy, pitch, vpitch, 
        z, vz, yaw, vyaw, 
        pos_ref, v_ref, mode
    ):
        pass

def sample_point(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.random.uniform(low, high) 

def run_ref(ref_state, time_step):
    return np.array([ref_state[0]+time_step*ref_state[1], ref_state[1]])

def get_bounding_box(hull: scipy.spatial.ConvexHull) -> np.ndarray:
    vertices = hull.points[hull.vertices, :]
    lower_bound = np.min(vertices, axis=0)
    upper_bound = np.max(vertices, axis=0)
    return np.vstack((lower_bound, upper_bound))

def sample_point_poly(hull: scipy.spatial.ConvexHull, n: int) -> np.ndarray:
    vertices = hull.points[hull.vertices,:]
    weights = np.random.uniform(0,1,vertices.shape[0])
    weights = weights/np.sum(weights)
    start_point = np.zeros(vertices.shape[1])
    for i in range(vertices.shape[1]):
        start_point[i] = np.sum(vertices[:,i]*weights)
    # return start_point

    sampled_point = []
    for i in range(n):
        vertex_idx = np.random.choice(hull.vertices)
        vertex = hull.points[vertex_idx, :]
        offset = vertex - start_point 
        start_point = start_point + np.random.uniform(0,1)*offset 
        sampled_point.append(start_point)

    return np.array(sampled_point)

def get_next_poly(trace_list) -> scipy.spatial.ConvexHull:
    vertex_list = []
    sample_vertex = np.zeros((0,6))
    for trace in trace_list:
        # trace = pickle.loads(tmp)
        rect_low = trace[-2,1:7]
        rect_high = trace[-1,1:7]
        # tmp = [
        #     [rect_low[0], rect_high[0]],
        #     [rect_low[1], rect_high[1]],
        #     [rect_low[2], rect_high[2]],
        #     [rect_low[3], rect_high[3]],
        #     [rect_low[4], rect_high[4]],
        #     [rect_low[5], rect_high[5]],
        # ]
        # vertices = np.array(list(itertools.product(*tmp)))
        sample = np.random.uniform(rect_low, rect_high)
        sample_vertex = np.vstack((sample_vertex, sample))
        vertex_list.append(rect_low)
        vertex_list.append(rect_high)

    # sample_idx  = np.random.choice(sample_vertex.shape[0], sample_vertex.shape[0]-23, replace = False)
    # sample_vertex = sample_vertex[sample_idx, :]
    sample_vertex = sample_vertex[:64,:]
    vertices = []
    for vertex in vertex_list:
        away = True
        for i in range(len(vertices)):
            if np.linalg.norm(np.array(vertex)-np.array(vertices[i]))<0.05:
                away = False
                break 
        if away:
            vertices.append(vertex)

    vertices = np.array(vertices)
    hull = scipy.spatial.ConvexHull(vertices, qhull_options='Qx Qt QbB Q12 Qc')    
    return hull, sample_vertex

def get_vision_estimation(point, M):
    estimate_low = point - np.array([0.01, 0, 0.01, 0, 0.01, 0, 0.01, 0, 0.01, 0, 0.01, 0])
    estimate_high = point + np.array([0.01, 0, 0.01, 0, 0.01, 0, 0.01, 0, 0.01, 0, 0.01, 0])
    estimate_low = point 
    estimate_high = point
    return estimate_low, estimate_high

def verify_step(point, M, computation_steps, time_steps, ref):
    # print(C_step, step, i, point)

    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False, reachability_method=ReachabilityMethod.DRYVR_DISC)) 
    aircraft = DroneAgent("a1")
    fixed_wing_scenario.add_agent(aircraft)
    estimate_low, estimate_high = get_vision_estimation(point, M)
    init_low = np.concatenate((point, estimate_low, ref))
    init_high = np.concatenate((point, estimate_high, ref))
    init = np.vstack((init_low, init_high))       

    fixed_wing_scenario.set_init(
        [init],
        [
            (DroneMode.Normal,)
        ],
    )
    # TODO: WE should be able to initialize each of the balls separately
    # this may be the cause for the VisibleDeprecationWarning
    # TODO: Longer term: We should initialize by writing expressions like "-2 \leq myball1.x \leq 5"
    # "-2 \leq myball1.x + myball2.x \leq 5"
    traces = fixed_wing_scenario.verify(computation_steps, time_steps, params={'bloating_method':'GLOBAL'})
    # tmp = pickle.dumps(np.array(traces))
    # tmp = pickle.dumps(traces.root.trace['a1'])
    return np.array(traces.root.trace['a1'])

def check_containment(post_rect, R, idx):
    return 'safe'

class SimTreeNode:
    def __init__(self, point, trace):
        self.trace = trace
        self.point = point 
        self.children = []

# tmp = 0

def compute_sim_tree_recur(drone_agent: DroneAgent, M, init_point, ref, num_sample_y, depth, C_compute_step , computation_steps, time_steps):
    global tmp
    full_point = np.concatenate((init_point, init_point, ref))
    node = SimTreeNode(full_point, None)  
    # print(tmp, depth)
    if depth < C_compute_step:
        lb_est, ub_est = get_vision_estimation(init_point, M)
        for i in range(num_sample_y):
            y_sample = sample_point(lb_est, ub_est)
            init_condition = np.concatenate((init_point, y_sample, ref))
            trace = drone_agent.TC_simulate(None, init_condition, computation_steps, time_steps)
            lp = trace[-1,1:13]
            new_ref = trace[-1,25:]
            child = compute_sim_tree_recur(drone_agent, M, lp, new_ref, num_sample_y, depth+1, C_compute_step, computation_steps, time_steps)
            node.children.append(child)
    return node

def compute_sim_tree(drone_agent, M, init_point, ref, num_sample_y, C_compute_step , computation_steps, time_steps):
    root = compute_sim_tree_recur(drone_agent, M, init_point, ref, num_sample_y, 0, C_compute_step, computation_steps, time_steps)
    return root 

def get_all_traces(node, depth, current_path = [], all_paths = []):
    if not node:
        return []
    
    current_path.append(np.insert(node.point,0, depth*0.1))

    if not node.children:
        all_paths.append(np.array(copy.deepcopy(current_path)))
    else:
        for child in node.children:
            get_all_traces(child, depth+1, current_path, all_paths)
    current_path.pop()
    return all_paths

def generate_trajectories(drone_agent, M, X_0, ref, num_sample_x, num_sample_y, C_compute_step, computation_steps, time_steps):
    # Sample points from X_0
    init_point_list = [sample_point(X_0[0,:], X_0[1,:]) for _ in range(num_sample_x)]

    # For each sampled points in X_0
    all_traces = []
    center = (X_0[0,:]+X_0[1,:])/2
    lb, ub = get_vision_estimation(center, X_0)
    init_set_low = np.concatenate((X_0[0,:], lb, ref))
    init_set_high = np.concatenate((X_0[1,:], ub, ref))
    init_set = np.vstack((init_set_low, init_set_high))
    root = compute_sim_tree_recur(drone_agent, M, center, ref, 1, 0, C_compute_step, computation_steps, time_steps)
    trace = get_all_traces(root,0)
    all_traces += trace
    for i, init_point in enumerate(init_point_list):
        print(i)
        root: SimTreeNode = compute_sim_tree_recur(drone_agent, M, init_point, ref, num_sample_y, 0, C_compute_step, computation_steps, time_steps)
        traces = get_all_traces(root, 0, current_path = [], all_paths = [])
        all_traces += traces 
    # Get corresponding perception contract values 
    # Sample points perception 
    # Compute simulation for one time step
    # Combine all results 
    # Return all results
    return all_traces, init_set

def compute_and_check(X_0, M, R, depth=0):
    # x, y, z, yaw, pitch, v
    # state = np.array([
    #     [-3050.0, -20, 110.0, 0-0.01, -np.deg2rad(3)-0.01, 10-0.1], 
    #     [-3010.0, 20, 130.0, 0+0.01, -np.deg2rad(3)+0.01, 10+0.1]
    # ])
    state = X_0

    # Parameters
    num_sample_x = 5
    num_sample_y = 1
    computation_steps = 0.1
    C_compute_step = 300
    C_num = 1
    parallel = True
    time_steps = 0.01
    
    ref = np.array([0, 1])

    C_list = [np.hstack((np.array([[0],[0]]),state))]
    # point_idx_list_list = []
    # point_list_list = []

    fig = pv.Plotter()
    for C_step in range(C_num):
        # try:
        # Generate simulation trajectories
        drone = DroneAgentPC('a1', M)
        init_set_low = np.concatenate((state[0,:], ref))
        init_set_high = np.concatenate((state[1,:], ref))
        init = np.vstack((init_set_low, init_set_high))
        # trajectories, init = generate_trajectories(drone, M, state, ref, num_sample_x, num_sample_y, C_compute_step, computation_steps, time_steps)

        # Compute reachable set by using the simulation trajecotries
        drone_scenario = Scenario(ScenarioConfig(parallel = False, reachability_method=ReachabilityMethod.DRYVR_DISC))
        drone_scenario.add_agent(drone)
        drone_scenario.set_init(
            [init],
            [
                (DroneMode.Normal,)
            ]
        )
        # Get rectangle 
        res = drone_scenario.verify(
            C_compute_step*computation_steps, 
            computation_steps, 
            params={
                'bloating_method':'GLOBAL',
                # 'traces':trajectories
            }
        )
        # fig = plot3dMap(tmp_map, ax=fig, width=0.05)
        fig = plot3dReachtube(res, "a1", 1, 5, 9, "r", fig, edge=True)
        # fig = plot3dReachtube(traces, "test2", 1, 2, 3, "b", fig, edge=True)
        # fig.set_background("#e0e0e0")
   
        # Check containment
        reachable_set = np.array(res.root.trace['a1'])
        next_init = reachable_set[-2:, 1:]
        C_set = np.hstack((np.array([[C_step+1],[C_step+1]]), next_init))
        res = check_containment(C_set, R, C_step+1)
        if res == 'unsafe' or res == 'unknown':
            return res, C_list

        C_list.append(C_set)
        state = next_init[:,:12] 
        ref = run_ref(ref, C_compute_step*computation_steps)

    fig.show()
    return 'safe', C_list
                

def visualize_outlier(num_outlier_list, E, idx = 0):
    full_e = np.ones((20, 14))*(-1)
    for i in range(len(E)):
        E_part = E[i]
        idx1 = round((E_part[0,0]-0.2)/0.05)
        idx2 = round((E_part[0,1]-(-0.1))/0.05)
        full_e[idx1, idx2] = num_outlier_list[i]
    full_e = full_e/np.max(full_e)
    rgba_image = np.zeros((20, 14, 4))  # 4 channels: R, G, B, A
    rgba_image[..., :3] = plt.cm.viridis(full_e)[..., :3]  # Apply a colormap    
    mask = np.where(full_e<0)
    rgba_image[..., 3] = 1.0  # Set alpha to 1 (non-transparent)

    # Apply the mask to make some pixels transparent
    rgba_image[mask[0], mask[1], :3] = 1  # Set alpha to 0 (transparent) for masked pixels

    plt.imshow(rgba_image)
    plt.savefig(f'./{idx}.png')

def refineEnv(E_in, M, data, id = 0, vis = False):
    E = copy.deepcopy(E_in)
    state_array, trace_array, E_array = data 

    dist_array = np.linalg.norm(state_array[:,:5]-trace_array[:,:5], axis=1)

    # Get environmental parameters of those points 
    percentile = np.percentile(dist_array, 90)

    # Remove those environmental parameters from E
    idx_list = np.where(dist_array>percentile)[0]

    E = np.array(E)
    num_outlier_list = np.zeros(E.shape[0])
    for idx in idx_list:
        contains = np.where(
            (E[:,0,0]<E_array[idx,0]) & \
            (E_array[idx,0]<E[:,1,0]) & \
            (E[:,0,1]<E_array[idx,1]) & \
            (E_array[idx,1]<E[:,1,1])
        )[0]
        num_outlier_list[contains] += 1
    sorted_outlier = np.argsort(num_outlier_list)
    delete_outlier = sorted_outlier[-10:] 
    if vis:
        visualize_outlier(num_outlier_list, E, id)
    E = np.delete(E, delete_outlier, axis=0)
    return E 

def refineState(X, idx):
    # TODO: Change refineState to meet the
    # requirement for this example 
    P1 = copy.deepcopy(X)
    P2 = copy.deepcopy(X)
    if idx%4==0 or idx%4==1:
        P1[1,1] = (P1[0,1] + P1[1,1])/2
        P2[0,1] = (P2[0,1] + P2[1,1])/2
    elif idx%4==2:
        P1[1,3] = (P1[0,3] + P1[1,3])/2
        P2[0,3] = (P2[0,3] + P2[1,3])/2
    elif idx%4==3:
        P1[1,0] = (P1[0,0] + P1[1,0])/2
        P2[0,0] = (P2[0,0] + P2[1,0])/2
    elif idx%4==4:
        P1[1,2] = (P1[0,2] + P1[1,2])/2
        P2[0,2] = (P2[0,2] + P2[1,2])/2
    elif idx%4==5:
        P1[1,4] = (P1[0,4] + P1[1,4])/2
        P2[0,4] = (P2[0,4] + P2[1,4])/2
    return P1, P2

def remove_data(data, E):
    # Remove data that are not in the range of E. 
    state_array, trace_array, E_array = data 
    idx = np.array([])
    for E_range in E:
        tmp = np.where((E_array[:,0]>E_range[0,0]) &\
                       (E_array[:,0]<E_range[1,0]) &\
                       (E_array[:,1]>E_range[0,1]) &\
                       (E_array[:,1]<E_range[1,1])
                       )[0]
        idx = np.concatenate((idx,tmp))
    idx = np.unique(idx).astype('int')
    return copy.deepcopy(state_array[idx,:]), copy.deepcopy(trace_array[idx,:]), copy.deepcopy(E_array[idx,:])

def computeContract(data, E):
    # Trim data according to the environments
    trimed_data = remove_data(data, E)
    M = get_all_models(trimed_data)
    return M

def findM(X_0, E_0, R, data):
    E = E_0 
    M = computeContract(data, E)
    queue = [(X_0,0)]
    while len(queue) != 0:
        P, idx = queue.pop(0)
        res, C_list = compute_and_check(P, M, R, idx)
        if res == "unsafe":
            print("unsafe")
            return None, None, P, C_list
        elif res == "unknown":
            print("unknown")
            if False:
                P_1, P_2 = refineState(P, idx)
                queue.append((P_1, idx+1))
                queue.append((P_2, idx+1))
            else:
                queue.append((P, idx+1))
            E = refineEnv(E, M, data, True)
            if len(E) <= 10:
                return None, None, P, C_list
            M = computeContract(data, E)
    print("safe")
    return M, E, None, C_list

if __name__ == "__main__":
    fn = os.path.join(script_dir, '../camera_path.json')
    with open(fn, 'r') as f:
        data = json.load(f)
    cam_data = data.get('camera_path')

    cam_states = np.zeros((len(cam_data),16))

    for j in range(len(cam_data)):
        cam_states[j] = np.array(cam_data[j].get('camera_to_world'))

    spline_fn = os.path.join(script_dir, '../camera_path_spline.json')    
    drone_agent = DroneAgent('drone',ref_spline=spline_fn)
    
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
    ref = np.array([0,1])
    num_sample_x = 5
    num_sample_y = 1
    C_compute_step = 20
    computation_steps = 0.1 
    time_steps = 0.01

    fn = os.path.join(script_dir, './exp2_train1.pickle')
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    
    M = get_all_models(data)

    res, C_list = compute_and_check(X_0, M, None,0)
    print(C_list)