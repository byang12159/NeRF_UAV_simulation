import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import copy
import dynamics3
from dynamics3 import g, A, B, kT
import controller
from controller import simulate, angle
import scipy
from scipy.integrate import odeint

# Initialization
N=300
mean = 0
covariance = np.diag([9,9])
std_dev = 3

random_x = np.random.normal(mean, std_dev, N)
random_y = np.random.normal(mean, std_dev, N)
random_z = np.random.normal(mean, std_dev, N)

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(random_x, random_y, random_z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
# Show the plot
plt.show()

############################# PREDEFINED TRAJECTORY ##################################
 # x, vx, theta_x, omega_x, y, vy, theta_y, omega_y, z, vz, theta_z, omega_z
def gen_traj():
    x0 = np.zeros(12)
    x0[10] = 0
    dt = 0.01 
    goal = np.array([5.,5.,5.,np.pi/2])
    x_list = [copy.deepcopy(x0)]

    i_hist = []
    statesx = []
    statesy = []
    statesz = []
    statesyaw = []
    statespitch = []
    statesroll = []
    for i in range(4000):
        res = simulate(x0, copy.deepcopy(goal), dt)
        x_list.append(copy.deepcopy(res))
        x0 = res
        statesx.append(x0[0])
        statesy.append(x0[4])
        statesz.append(x0[8])
        statesyaw.append(x0[10])
        statespitch.append(x0[6])
        statesroll.append(x0[2])

    statesu = np.zeros(len(statesx))
    statesv = np.zeros(len(statesx))
    statesw = np.zeros(len(statesx))

    for i in range(len(statespitch)):
        statesu[i],statesv[i],statesw[i] = angle(statesyaw[i],statespitch[i],statesroll[i])

    scale_down_factor = 200
    statesx = statesx[::scale_down_factor]
    statesy = statesy[::scale_down_factor]
    statesz = statesz[::scale_down_factor]

    statesu = statesu[::scale_down_factor]
    statesv = statesv[::scale_down_factor]
    statesw = statesw[::scale_down_factor]

    # # Create a figure and a 3D axis
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # Plot the data points as markers
    # # ax.scatter(statesx, statesy, statesz, c='r', marker='o')
    # ax.quiver(statesx, statesy, statesz, statesu, statesv, statesw, length=0.4, normalize=True)
    # # Label the axes
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # # Show the plot
    # plt.show()

    return np.column_stack((statesx,statesy,statesz))

traj_states = gen_traj() # Colmn of x,y,z
traj_step = 1

print("traj states",traj_states.shape)
gt = [0,0,0]
m = gt
gt_labels = [gt]
samples = np.column_stack((random_x,random_y,random_z))

sigma=1 # standard deviation
variance = sigma**2
def likelihood(location,m):
    return np.exp(-0.5*(location-m)**2/variance)

def move(samples):
    # Simulate all samples forward for one step
    V=2
    predictions = np.copy(samples)
    
    x = predictions[:,0]
    y = predictions[:,1]
    z = predictions[:,2]
    predictions[:,0] += traj_states[traj_step][0] - traj_states[traj_step-1][0]
    predictions[:,1] += traj_states[traj_step][1] - traj_states[traj_step-1][1]
    predictions[:,2] += traj_states[traj_step][2] - traj_states[traj_step-1][2]

    gt[0] += traj_states[traj_step][0] - traj_states[traj_step-1][0]
    gt[1] += traj_states[traj_step][1] - traj_states[traj_step-1][1]
    gt[2] += traj_states[traj_step][2] - traj_states[traj_step-1][2]

    return predictions,gt

for i in range(19):

    weights = np.apply_along_axis(likelihood, 1, samples, m)

    # Motion Update
    predictions, gt = move(samples)

    gt_labels.append(gt)
    m = gt

    # Resample
    sample_indices = np.random.choice(len(samples),p=(weights[:,0]+weights[:,1]+weights[:,2])/(np.sum(weights[:,0])+np.sum(weights[:,1])+np.sum(weights[:,2])),size=N)
    samples = predictions[sample_indices]


    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(samples[:,0],samples[:,1],samples[:,2],'*')
    ax.scatter(gt[0],gt[1],gt[2],'o',color = 'red')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(-5, 30)  # Set X-axis limits
    ax.set_ylim(-5, 30)  # Set Y-axis limits
    ax.set_zlim(-5, 30)  # Set Z-axis limits
    # Show the plot
    plt.show()

