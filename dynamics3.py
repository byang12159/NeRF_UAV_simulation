### From https://arxiv.org/pdf/1703.07373.pdf Eq. (19) (Left)
import numpy as np
from numpy import cos, sin

# quadrotor physical constants
g = 9.81; d0 = 10; d1 = 8; n0 = 10; kT = 0.91

# non-linear dynamics
def f(state, u):
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
# A = np.zeros([12, 12])
# A[0, 1] = 1.
# A[1, 2] = g
# A[2, 2] = -d1
# A[2, 3] = 1
# A[3, 2] = -d0
# A[4, 5] = 1.
# A[5, 6] = g
# A[6, 6] = -d1
# A[6, 7] = 1
# A[7, 6] = -d0
# A[8, 9] = 1.
# A[10, 11] = 1

# B = np.zeros([12, 4])
# B[3, 0] = n0
# B[7, 1] = n0
# B[9, 2] = kT
# B[11,3] = n0

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