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
import open3d as o3d

from z3 import *

script_dir = os.path.realpath(os.path.dirname(__file__))

# def apply_model_batch(model, point):
#     cc = model['coef_center']
#     cr = model['coef_radius']

#     x = point[0]
#     y = point[4]
#     z = point[8]
#     c = cc[0] + cc[1]*x + cc[2]*y + cc[3]*z 
#     r = cr[0] + cr[1]*x + cr[2]*y + cr[3]*z 
#     return c, abs(r)

def apply_model(model, point):
    cc = model['coef_center']
    cr = model['coef_radius']
    dim = model['dim']
    
    if dim=='x' or dim=='y' or dim=='z':
        x = (point.T)[0]
        y = (point.T)[1]
        z = (point.T)[2]
        c = cc[0] + cc[1]*x + cc[2]*y + cc[3]*z 
        r = cr[0] + cr[1]*x + cr[2]*y + cr[3]*z 
    else:
        x = (point.T)[0]
        y = (point.T)[1]
        z = (point.T)[2]
        if dim=='roll':
            w = (point.T)[3]
        elif dim=='pitch':
            w = (point.T)[4]
        else:
            w = (point.T)[5]
        c = cc[0] + cc[1]*x + cc[2]*y + cc[3]*z + cc[4]*w 
        r = cr[0] + cr[1]*x + cr[2]*y + cr[3]*z + cr[4]*w
        
    return c, abs(r)

def get_vision_estimation_batch(point, models):
    x_c, x_r = apply_model(models[0], point)
    y_c, y_r = apply_model(models[1], point)
    z_c, z_r = apply_model(models[2], point)
    yaw_c, yaw_r = apply_model(models[5], point)    
    low = np.array([x_c-x_r, y_c-y_r, z_c-z_r, yaw_c-yaw_r])
    high = np.array([x_c+x_r, y_c+y_r, z_c+z_r, yaw_c+yaw_r])

    return low, high 

def create_box(box):
                
    translation = box[3:6]
    h, w, l = box[0], box[1], box[2]

    rotationz = box[6]
    rotationy = box[7]
    rotationx = box[8]

    # Create a bounding box outline if x,y,z is center point
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])
    
    # Create a bounding box outline if x,y,z is rear center point
    # bounding_box = np.array([
    #             [l,l,0,0,l,l,0,0],                      
    #             [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2],          
    #             [0,0,0,0,h,h,h,h]])                        

    rotation_matrixZ = np.array([
        [np.cos(rotationz), -np.sin(rotationz), 0.0],
        [np.sin(rotationz), np.cos(rotationz), 0.0],
        [0.0, 0.0, 1.0]])
    
    rotation_matrixY = np.array([
        [np.cos(rotationy), 0.0, np.sin(rotationy)],
        [0.0, 1.0 , 0.0],
        [-np.sin(rotationy), 0.0,  np.cos(rotationy)]])

    rotation_matrixX = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(rotationx), -np.sin(rotationx)],
        [0.0, np.sin(rotationx),  np.cos(rotationx)]])
    
    
    rotation_matrix = rotation_matrixZ@rotation_matrixY@rotation_matrixX
    # print(rotation_matrix)
    # Repeat the center position [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = rotation_matrix@bounding_box + eight_points.transpose()
    return corner_box.transpose()


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

def check_containment(post_rect, idx):
    # return 'safe'
    s = Solver()
    x,y,z = Reals('x y z')
    # High gate
    c_high = And(
        -3.724*x-0.4807<=y, 
        y<=-3.724*x-0.4653, 
        -0.025-0.1 <= z,
        z <= 0.091+0.1,
        -0.251-0.1 <=x, 
        x<=  -0.198+0.1,
        Or(
            z<=-0.017266, 
            z>=0.083266, 
            x<=-0.24326666667, 
            x>=-0.2057
        )
    )
    # Middle Gate 
    c_mid = And(
        -0.123*x + 0.0352 <= y,
        y <= -0.123*x+0.05073,
        -0.084-0.1 <= z, 
        z<= 0.035+0.1,
        0.271-0.1 <= x,
        x<= 0.472+0.1,
        Or(
            z <= -0.07626 ,
            z >= 0.027266 ,
            x<= 0.27873 ,
            x>= 0.46426
        )
    )
    # Low Gate 
    c_low = And(
        1.966*x-0.0127 <= y, 
        y <=  1.966*x+0.0027,
        -0.125-0.1  <= z,
        z <= -0.017+0.1,
        -0.239-0.1 <= x, 
        x<=  -0.147+0.1,
        Or(
            z <= -0.1172,
            z >= -0.024733,
            x<= -0.2313,
            x>= -0.1547
        )
    )
    s.add(Or(c_low, c_mid, c_high))
    for i in range(0, len(post_rect), 2):
        low = post_rect[i]
        high = post_rect[i+1]
        s.push()
        s.add(
            x>=low[1], x<=high[1],
            y>=low[5], y<=high[5],
            z>=low[9], z<=high[9]
        )
        if s.check()==sat:
            print('result unknown')
            return 'unknown'
        s.pop()
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

# def compensate(r):
#     r[::2, 1] -= 0.005
#     r[1::2, 1] += 0.005
#     r[::2, 5] -= 0.005
#     r[1::2, 5] += 0.005
#     r[::2, 9] -= 0.01
#     r[1::2, 9] += 0.01
#     return r


def compute_and_check(X_0, M, R, depth=0, computation_steps = 0.1, C_compute_step = 10, C_num = 30):
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
    C_compute_step = 20    
    C_num = 15
    parallel = True
    time_steps = 0.01
    
    ref = np.array([0, 1])

    C_list = [np.hstack((np.array([[0],[0]]),state))]
    reachtube = None
    # point_idx_list_list = []
    # point_list_list = []

    # fig = pv.Plotter()
    for C_step in range(C_num):
        print(C_step)
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
                'sim_trace_num': 55
                # 'traces':trajectories
            }
        )
        # fig = plot3dMap(tmp_map, ax=fig, width=0.05)
        # fig = plot3dReachtube(res, "a1", 1, 5, 9, "r", fig, edge=True)
        # fig = plot3dReachtube(traces, "test2", 1, 2, 3, "b", fig, edge=True)
        # fig.set_background("#e0e0e0")
   
        # Check containment
        reachable_set = np.array(res.root.trace['a1'])
        next_init = copy.deepcopy(reachable_set[-2:, 1:])

        # reachable_set = compensate(reachable_set)
        # reachable_set[::2, 9] -= 0.01
        # reachable_set[1::2, 9] += 0.01

        if reachtube is None: 
            reachtube = reachable_set
        else:
            reachtube = np.vstack((reachtube, reachable_set))
        C_set = np.hstack((np.array([[C_step+1],[C_step+1]]), next_init))

        res = check_containment(reachable_set, C_step)
        if res == 'unsafe' or res == 'unknown':
            return res, C_list, reachtube

        C_list.append(C_set)
        state = next_init[:,:12] 
        ref = run_ref(ref, C_compute_step*computation_steps)

    # fig.show()
    return 'safe', C_list, reachtube
                

def visualize_outlier(num_outlier_list, E, idx = 0):
    full_e = np.ones((10, 10))*(-1)
    for i in range(len(E)):
        E_part = E[i]
        idx1 = round((E_part[0,0]-0)/0.1)
        idx2 = round((E_part[0,1]-0)/0.1)
        full_e[idx1, idx2] = num_outlier_list[i]
    full_e = full_e/np.max(full_e)
    rgba_image = np.zeros((10, 10, 4))  # 4 channels: R, G, B, A
    rgba_image[..., :3] = plt.cm.viridis(full_e)[..., :3]  # Apply a colormap    
    mask = np.where(full_e<0)
    rgba_image[..., 3] = 1.0  # Set alpha to 1 (non-transparent)

    # Apply the mask to make some pixels transparent
    rgba_image[mask[0], mask[1], :3] = 1  # Set alpha to 0 (transparent) for masked pixels

    plt.imshow(rgba_image)
    plt.savefig(f'./{idx}.png')

def refineEnv(E_in, M, data, id = 0, vis = False):
    E = copy.deepcopy(E_in)

    E_center_list = []
    for Ep in E:
        E_center = (Ep[0,:]+Ep[1,:])/2
        E_center_list.append(np.linalg.norm(E_center))

    sorted_outlier = np.argsort(E_center_list)
    delete_outlier = sorted_outlier[-4:] 
    E = np.delete(E, delete_outlier, axis=0)

    # state_array, trace_array, E_array = data 

    # dist_array = np.linalg.norm(state_array[:,:5]-trace_array[:,:5], axis=1)

    # # Get environmental parameters of those points 
    # percentile = np.percentile(dist_array, 90)

    # # Remove those environmental parameters from E
    # idx_list = np.where(dist_array>percentile)[0]

    # E = np.array(E)
    # num_outlier_list = np.zeros(E.shape[0])
    # for idx in idx_list:
    #     contains = np.where(
    #         (E[:,0,0]<=E_array[idx,0]) & \
    #         (E_array[idx,0]<=E[:,1,0]) & \
    #         (E[:,0,1]<=E_array[idx,1]) & \
    #         (E_array[idx,1]<=E[:,1,1])
    #     )[0]
    #     num_outlier_list[contains] += 1
    # sorted_outlier = np.argsort(num_outlier_list)
    # delete_outlier = sorted_outlier[-10:] 
    # if vis:
    #     visualize_outlier(num_outlier_list, E, id)
    # E = np.delete(E, delete_outlier, axis=0)
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
        tmp = np.where((E_array[:,0]>=E_range[0,0]) &\
                       (E_array[:,0]<=E_range[1,0]) &\
                       (E_array[:,1]>=E_range[0,1]) &\
                       (E_array[:,1]<=E_range[1,1])
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
    idx = 0
    while len(queue) != 0:
        P, idx = queue.pop(0)
        res, C_list, reachtube = compute_and_check(P, M, R, idx)
        print(f">>>>> {idx}")
        if res == "unsafe":
            print("unsafe")
            return None, E, P, C_list, reachtube
        elif res == "unknown":
            print("unknown")
            if False:
                P_1, P_2 = refineState(P, idx)
                queue.append((P_1, idx+1))
                queue.append((P_2, idx+1))
            else:
                queue.append((P, idx+1))
            E = refineEnv(E, M, data, True)
            if len(E) <= 2:
                return None, E, P, C_list, reachtube
            M = computeContract(data, E)
    print(idx)
    print("safe")
    return M, E, None, C_list, reachtube

def partitionE(E):
    partition_list = []
    E1 = np.arange(E[0,0], E[1,0]+0.01, 0.1)
    E2 = np.arange(E[0,1], E[1,1]+0.01, 0.1)

    for i in range(len(E1)-1):
        for j in range(len(E2)-1):
            partition = np.array([
                [E1[i], E2[j]],
                [E1[i+1], E2[j+1]]
            ])
            partition_list.append(partition)

    return partition_list

def test_accuracy(data, M):
    state_array, est_array, E_array = data 
    total_data = 0
    data_satisfy = 0
    notin0 = 0
    notin1 = 0
    notin2 = 0
    notin3 = 0
    notin4 = 0
    notin5 = 0
    # Get accuracy of perception contract
    for j in range(len(state_array)):
        gt = np.array(state_array[j])
        est = np.array(est_array[j])

        cx, rx = apply_model(M[0], gt)
        cy, ry = apply_model(M[1], gt)
        cz, rz = apply_model(M[2], gt)
        croll, rroll = apply_model(M[3], gt)
        cpitch, rpitch = apply_model(M[4], gt)
        cyaw, ryaw = apply_model(M[5], gt)
        

        total_data += 1
        if (est[0]>=cx-rx and est[0]<=cx+rx and \
            est[1]>=cy-ry and est[1]<=cy+ry and \
            est[2]>=cz-rz and est[2]<=cz+rz and \
            est[3]>=croll-rroll and est[3]<=croll+rroll and \
            est[4]>=cpitch-rpitch and est[4]<=cpitch+rpitch and \
            est[5]>=cyaw-ryaw and est[5]<=cyaw+ryaw
            ):
            data_satisfy += 1 
        else:
            if not (est[0]>=cx-rx and est[0]<=cx+rx):
                notin0 += 1 
            if not (est[1]>=cy-ry and est[1]<=cy+ry):
                notin1 += 1                
            if not (est[2]>=cz-rz and est[2]<=cz+rz):
                notin2 += 1
            if not (est[3]>=croll-rroll and est[3]<=croll+rroll):
                notin3 += 1
            if not (est[4]>=cpitch-rpitch and est[4]<=cpitch+rpitch):
                notin4 += 1
            if not (est[5]>=cyaw-ryaw and est[5]<=cyaw+ryaw):
                notin5 += 1
    print(data_satisfy/total_data)
    print(total_data, notin0, notin1, notin2, notin3, notin4, notin5)           

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
    computation_steps = 0.1
    C_compute_step = 300
    C_num = 1
    time_steps = 0.01

    E = np.array([
        [0.0, -1.0],
        [1.0, 0.0]
    ])

    E = partitionE(E)

    E = np.array([[
        [0, -0.1],
        [0.1, 0]
    ]])

    fn = os.path.join(script_dir, './exp2_train4.pickle')
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    
    # for i in range(9):
    #     E = refineEnv(E, None, data, i)
    # E = np.array([
    #     [[0, -0.1],
    #     [0.1, 0.0]],
    #     # [[0, -0.2],
    #     #  [0.1, -0.1]],
    #     # [[0, -0.3],
    #     #  [0.1, -0.2]],  
    # ])
    # for i in range(2):
    #     for j in range(2):
    #         Ep = np.array([
    #             [i*0.1, (j+1)*(-0.1)],
    #             [(i+1)*0.1, j*(-0.1)]
    #         ])
    #         if i==0 and j==0:
    #             E = Ep.reshape((1,2,2))
    #         else:
    #             E = np.concatenate((E, Ep.reshape((1,2,2))), axis=0)
    # M = computeContract(data, E)

    # res, C_list, reachtube = compute_and_check(X_0, M, None,0, computation_steps, C_compute_step, C_num)
    M, E, Xc, C_list, reachtube = findM(X_0, E, None, data)
    # print(C_list)
    with open(os.path.join(script_dir, 'exp2_safe_small.pickle'), 'wb+') as f: 
        pickle.dump((M, E, C_list, reachtube), f)

    # Load the .ply file
    pc_fn = os.path.join(script_dir, 'point_cloud.ply')
    point_cloud = o3d.io.read_point_cloud(pc_fn)
    # Create a red point at the center of the coordinate system
    center_point = np.array([[0.0, 0.0, 0.0]]) # Center point coordinates
    center_color = np.array([[1.0, 0.0, 0.0]]) # Red color (R,G,B)
    point1 = np.array([[0.46990477000436776,-0.01279251651773311,0.02605140075172408]])
    point2 = np.array([[-0.14787864315441807,-0.2924556311040014,-0.017899417569988384]])
    # Create coordinate frame axes
    coordinate_axes = o3d.geometry.LineSet()
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    lines = np.array([[0, 1], [0, 2], [0, 3]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # RGB colors for X, Y, Z axes
    coordinate_axes.points = o3d.utility.Vector3dVector(vertices)
    coordinate_axes.lines = o3d.utility.Vector2iVector(lines)
    coordinate_axes.colors = o3d.utility.Vector3dVector(colors)
    # Create a point cloud for the center point
    center_cloud = o3d.geometry.PointCloud()
    center_cloud.points = o3d.utility.Vector3dVector(center_point)
    center_cloud.colors = o3d.utility.Vector3dVector(center_color)
    point1_cloud = o3d.geometry.PointCloud()
    point1_cloud.points = o3d.utility.Vector3dVector(point1)
    point1_cloud.colors = o3d.utility.Vector3dVector(np.array([[0.0, 1.0, 0.0]]))
    point2_cloud = o3d.geometry.PointCloud()
    point2_cloud.points = o3d.utility.Vector3dVector(point2)
    point2_cloud.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0]]))
    # Create a visualization window
    object_list = [point_cloud, center_cloud, point1_cloud, point2_cloud, coordinate_axes]
    for i in range(0, reachtube.shape[0], 2):
        lb = reachtube[i,[1,5,9]]
        ub = reachtube[i+1,[1,5,9]]
        l, w, h = ub-lb 
        x,y,z = (lb+ub)/2 
        box = [h,w,l,x,y,z,0,0,0]
        boxes3d_pts = create_box(box)
        boxes3d_pts = o3d.utility.Vector3dVector(boxes3d_pts)
        box = o3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts)
        box.color = [1, 0, 0]  #Box color would be red box.color = [R,G,B]
        object_list.append(box)

        # box = [h,w,l,x,y,z,0,0,0]
        # boxes3d_pts = create_box(box)
        # boxes3d_pts = o3d.utility.Vector3dVector(boxes3d_pts)
        # box = o3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts)
        # box.color = [0, 0, 1]  #Box color would be red box.color = [R,G,B]
        
        # object_list.append(box)

    o3d.visualization.draw_geometries(object_list, window_name="Point Cloud with Axes", width=800, height=600)
