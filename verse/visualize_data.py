import pickle
import matplotlib.pyplot as plt 
import os
from get_all_models import get_all_models
from drone2 import computeContract, refineEnv, partitionE, apply_model
import numpy as np 
from typing import Tuple 

script_dir = os.path.dirname(os.path.realpath(__file__))


def get_vision_estimation(point: np.ndarray, models) -> Tuple[np.ndarray, np.ndarray]:
    x_c, x_r = apply_model(models[0], point)
    y_c, y_r = apply_model(models[1], point)
    z_c, z_r = apply_model(models[2], point)
    roll_c, roll_r = apply_model(models[3], point)
    pitch_c, pitch_r = apply_model(models[4], point)
    yaw_c, yaw_r = apply_model(models[5], point)
    
    low = np.zeros((len(x_c), 6))
    # low = np.array([
    #     x_c-x_r, point[1], point[2], point[3], 
    #     y_c-y_r, point[5], point[6], point[7], 
    #     z_c-z_r, point[9], point[10], point[11]
    # ])
    low[:,0] = x_c-x_r 
    low[:,1] = y_c-y_r 
    low[:,2] = z_c-z_r
    low[:,3] = roll_c-roll_r 
    low[:,4] = pitch_c-pitch_r 
    low[:,5] = yaw_c-yaw_r
    
    high = np.zeros((len(x_c), 6))
    high[:,0] = x_c+x_r 
    high[:,1] = y_c+y_r 
    high[:,2] = z_c+z_r
    high[:,3] = roll_c+roll_r 
    high[:,4] = pitch_c+pitch_r 
    high[:,5] = yaw_c+yaw_r

    return low, high

with open(os.path.join(script_dir, 'exp2_train4.pickle'), 'rb') as f:
    data = pickle.load(f)

state_array, trace_array, env_array = data 

E = np.array([
    [-0.1, -1.1],
    [1.1, 0.1]
])
E = partitionE(E)

M = computeContract(data, E)

lb1, ub1 = get_vision_estimation(state_array, M)

for i in range(12):
    E = refineEnv(E, None, data, i)

M = computeContract(data, E)
lb2, ub2 = get_vision_estimation(state_array, M)

plt.figure(0)
plt.plot(state_array[:,0], trace_array[:,0], '*')
plt.plot(state_array[:,0], lb1[:,0], 'r*')
plt.plot(state_array[:,0], ub1[:,0], 'r*')
plt.plot(state_array[:,0], lb2[:,0], 'g*')
plt.plot(state_array[:,0], ub2[:,0], 'g*')
plt.title('x')

plt.figure(1)
plt.plot(state_array[:,1], trace_array[:,1], '*')
plt.plot(state_array[:,1], lb1[:,1], 'r*')
plt.plot(state_array[:,1], ub1[:,1], 'r*')
plt.plot(state_array[:,1], lb2[:,1], 'g*')
plt.plot(state_array[:,1], ub2[:,1], 'g*')
plt.title('y')

plt.figure(2)
plt.plot(state_array[:,2], trace_array[:,2], '*')
plt.plot(state_array[:,2], lb1[:,2], 'r*')
plt.plot(state_array[:,2], ub1[:,2], 'r*')
plt.plot(state_array[:,2], lb2[:,2], 'g*')
plt.plot(state_array[:,2], ub2[:,2], 'g*')
plt.title('z')

plt.figure(3)
plt.plot(state_array[:,3], trace_array[:,3], '*')
plt.plot(state_array[:,3], lb1[:,3], 'r*')
plt.plot(state_array[:,3], ub1[:,3], 'r*')
plt.plot(state_array[:,3], lb2[:,3], 'g*')
plt.plot(state_array[:,3], ub2[:,3], 'g*')
plt.title('roll')

plt.figure(4)
plt.plot(state_array[:,4], trace_array[:,4], '*')
plt.plot(state_array[:,4], lb1[:,4], 'r*')
plt.plot(state_array[:,4], ub1[:,4], 'r*')
plt.plot(state_array[:,4], lb2[:,4], 'g*')
plt.plot(state_array[:,4], ub2[:,4], 'g*')
plt.title('pitch')

plt.figure(5)
plt.plot(state_array[:,5], trace_array[:,5], '*')
plt.plot(state_array[:,5], lb1[:,5], 'r*')
plt.plot(state_array[:,5], ub1[:,5], 'r*')
plt.plot(state_array[:,5], lb2[:,5], 'g*')
plt.plot(state_array[:,5], ub2[:,5], 'g*')
plt.title('yaw')



plt.show()