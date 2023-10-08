from verse.plotter.plotter2D import * 

from drone_agent import DroneAgent
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

import pickle 
import json 
import ray

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
    x_est: float 
    vx_est: float 
    roll_est: float 
    vroll_est: float 
    y_est: float 
    vy_est: float 
    pitch_est: float 
    vpitch_est: float 
    z_est: float 
    vz_est: float 
    yaw_est: float 
    vyaw_est: float 
    pos_ref: float 
    v_ref: float 
    mode: DroneMode 

    def __init__(self, 
        x, vx, roll, vroll, 
        y, vy, pitch, vpitch, 
        z, vz, yaw, vyaw, 
        x_est, vx_est, roll_est, vroll_est, 
        y_est, vy_est, pitch_est, vpitch_est, 
        z_est, vz_est, yaw_est, vyaw_est, 
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

def get_vertex_samples(hull):
    dim_ranges = []
    box = get_bounding_box(hull)
    important_dim = [0,2,4,6,8,10]
    box_important = box[:, important_dim]
    tmp = box_important.T.tolist()
    vertices = np.array(list(itertools.product(*tmp)))
    for i in range(len(vertices)):
        pass

def compute_and_check(X_0, M, R, depth=0):
    # x, y, z, yaw, pitch, v
    ray.init(num_cpus=12,log_to_driver=False)
    # state = np.array([
    #     [-3050.0, -20, 110.0, 0-0.01, -np.deg2rad(3)-0.01, 10-0.1], 
    #     [-3010.0, 20, 130.0, 0+0.01, -np.deg2rad(3)+0.01, 10+0.1]
    # ])
    state = X_0
    # tmp = [
    #     [state[0,0], state[1,0]],
    #     [state[0,1], state[1,1]],
    #     [state[0,2], state[1,2]],
    #     [state[0,3], state[1,3]],
    #     [state[0,4], state[1,4]],
    #     [state[0,5], state[1,5]],
    # ]
    tmp = state.T.tolist()
    vertices = np.array(list(itertools.product(*tmp)))
    hull = scipy.spatial.ConvexHull(vertices)
    # state_low = state[0,:]
    # state_high = state[1,:]

    # Parameters
    num_sample = 100
    computation_steps = 0.1
    C_compute_step = 80
    C_num = 10
    parallel = True
    time_steps = 0.01
    
    ref = np.array([0, 0.1])

    C_list = [np.hstack((np.array([[0],[0]]),state))]
    # point_idx_list_list = []
    # point_list_list = []

    for C_step in range(C_num):
        # try:
            reachable_set = []
            for step in range(C_compute_step):
                print(">>>>>>>>>>>>>>>>", depth, C_step, step)
                box = get_bounding_box(hull)
                state_low = box[0,:]
                state_high = box[1,:]

                reachable_set.append([np.insert(state_low, 0, step*computation_steps), np.insert(state_high, 0, step*computation_steps)])

                traces_list = []
                if step == 0:
                    # vertex_num = int(num_sample*0.05)
                    # sample_num = num_sample - vertex_num
                    # vertex_idxs = np.random.choice(hull.vertices, vertex_num)
                    vertex_sample = get_vertex_samples(hull)
                    # edge_sample = get_edge_samples(vertex_sample)
                    sample_sample = sample_point_poly(hull, num_sample)
                    samples = np.vstack((vertex_sample, sample_sample))
                else:
                    sample_sample = sample_point_poly(hull, num_sample)
                    samples = np.vstack((vertex_sample, sample_sample))
                    # samples = vertex_sample
                
                point_idx = np.argmax(hull.points[:,1])
                samples = np.vstack((samples, hull.points[point_idx,:]))
                # samples = sample_point_poly(hull, num_sample)
                
                point_idx = np.argmax(hull.points[:,0])
                samples = np.vstack((samples, hull.points[point_idx,:]))
                point_idx = np.argmin(hull.points[:,0])
                samples = np.vstack((samples, hull.points[point_idx,:]))

                task_list = []
                traces_list = []
                for i in range(samples.shape[0]):

                    point = samples[i,:]
                    
                    if parallel:
                        task_list.append(verify_step_remote.remote(point, M, computation_steps, time_steps, ref))
                    else:
                        print(C_step, step, i, point)
                        trace = verify_step(point, M, computation_steps, time_steps, ref)
                        traces_list.append(trace)

                if parallel:
                    traces_list = ray.get(task_list)
                # hull2, vertex_sample2 = get_next_poly(traces_list)
                hull, vertex_sample = get_next_poly(traces_list)
                # box1 = get_bounding_box(hull)
                # box2 = get_bounding_box(hull2)
                # if (box1 != box2).any():
                #     print('stop')
                # plt.figure(6)
                # plt.plot([step*0.1, step*0.1],[box[0,-1],box[1,-1]],'g')
                # state_low = next_low 
                # state_high = next_high 
                ref = run_ref(ref, computation_steps)
            
            next_init = get_bounding_box(hull)
            # last_rect = reachable_set[-1]
            # next_init = np.array(last_rect)[:,1:]
            C_set = np.hstack((np.array([[C_step+1],[C_step+1]]), next_init))
            
            # TODO: Check containment of C_set and R
            C_set[0,2] -= 1
            C_set[1,2] += 1
            C_set[0,3] -= 1
            C_set[1,3] += 1
            res = check_containment(C_set, R, C_step+1)
            if res == 'unsafe' or res == 'unknown':
                ray.shutdown()
                return res, C_list

            C_list.append(C_set)

            tmp = [
                [next_init[0,0], next_init[1,0]],
                [next_init[0,1], next_init[1,1]],
                [next_init[0,2], next_init[1,2]],
                [next_init[0,3], next_init[1,3]],
                [next_init[0,4], next_init[1,4]],
                [next_init[0,5], next_init[1,5]],
            ]
            vertices = np.array(list(itertools.product(*tmp)))
            hull = scipy.spatial.ConvexHull(vertices)
        # except:
        #     break

    ray.shutdown()
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
 