import numpy as np

x0 = np.zeros(12)
x0[10] = np.pi/3

print(x0)

curr_position = np.array(x0)[[0, 4, 8]]

print(curr_position)