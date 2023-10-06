import dynamics3
from dynamics3 import g, A, B, kT
import numpy as np
import scipy
from scipy.integrate import odeint
import copy 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

class Controller:
    def __init__(self):

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
        Q[0, 0] = 10.
        Q[1, 1] = 10.
        Q[2, 2] = 10.
        # Q[11,11] = 0.01
        R = np.diag([1., 1., 1.])
        self.K, _, _ = lqr(A, B, Q, R)
        # print(self.K)

    ####################### The controller ######################
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
        u_ori = (goal[3]-yaw)*0.1+(0-x[11])*1.0

        return np.concatenate((u_pos, [u_ori]))

    ######################## The closed_loop system #######################
    def cl_nonlinear(self, x, t, goal):
        x = np.array(x)
        dot_x = dynamics3.f(x, self.u(x, goal))
        return dot_x

    # simulate
    def simulate(self, x, goal, dt):
        curr_position = np.array(x)[[0, 4, 8]]
        goal_pos = goal[:3]
        error = goal_pos - curr_position
        distance = np.sqrt((error**2).sum())
        if distance > 1:
            goal[:3] = curr_position + error / distance
        return odeint(self.cl_nonlinear, x, [0, dt], args=(goal,))[-1]

    def angle(self, yaw,pitch,roll):
        # convert yaw, pitch, and roll angles (Euler angles, radian) to a 3D vector in global coordinates (u, v, w)
        # Create rotation matrices
        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

        R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                            [0, 1, 0],
                            [-np.sin(pitch), 0, np.cos(pitch)]])

        R_roll = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])

        # Combine the rotation matrices
        R_combined = np.dot(R_yaw, np.dot(R_pitch, R_roll))

        # Reference vector along the X-axis
        reference_vector_X = np.array([1, 0, 0])

        # Apply the combined rotation to the reference vector
        uvw_vector = np.dot(R_combined, reference_vector_X)
        return uvw_vector[0],uvw_vector[1],uvw_vector[2]

if __name__ == "__main__":
    ctr1 = Controller()
    
    # x, vx, theta_x, omega_x, y, vy, theta_y, omega_y, z, vz, theta_z, omega_z
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
        res = ctr1.simulate(x0, copy.deepcopy(goal), dt)
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
        statesu[i],statesv[i],statesw[i] = ctr1.angle(statesyaw[i],statespitch[i],statesroll[i])

    scale_down_factor = 30
    statesx = statesx[::scale_down_factor]
    statesy = statesy[::scale_down_factor]
    statesz = statesz[::scale_down_factor]

    statesu = statesu[::scale_down_factor]
    statesv = statesv[::scale_down_factor]
    statesw = statesw[::scale_down_factor]

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data points as markers
    ax.quiver(statesx, statesy, statesz, statesu, statesv, statesw, length=0.4, normalize=True)

    # Label the axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show()









