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
