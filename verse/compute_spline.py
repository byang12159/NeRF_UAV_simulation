import numpy as np 
import json 
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline, BSpline
import matplotlib.pyplot as plt
import os 

script_dir = os.path.dirname(os.path.realpath(__file__))
camera_path = os.path.join(script_dir, 'camera_path.json')

with open(camera_path, 'r') as file:
    data = json.load(file)

cam_data = data.get('camera_path')
cam_states = np.zeros((len(cam_data),16))

for j in range(len(cam_data)):
    cam_states[j] = cam_data[j].get('camera_to_world')

cam_states = cam_states.reshape((-1,4,4))
x = cam_states[:,0,3]
y = cam_states[:,1,3]
z = cam_states[:,2,3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, color='b')

t = [0]
for i in range(1, x.shape[0]):
    t.append(t[-1]+np.linalg.norm([x[i]-x[i-1], y[i]-y[i-1], z[i]-z[i-1]])*10)

spline_x = UnivariateSpline(t, x, k=3,s=0)
spline_y = UnivariateSpline(t, y, k=3,s=0)
spline_z = UnivariateSpline(t, z, k=3,s=0)

tck_x = spline_x._eval_args
tck_y = spline_y._eval_args
tck_z = spline_z._eval_args

res = {
    "x":{
        '0':tck_x[0].tolist(),
        '1':tck_x[1].tolist(),
        '2':tck_x[2]
    },
    "y":{
        '0':tck_y[0].tolist(),
        '1':tck_y[1].tolist(),
        '2':tck_y[2]
    },
    "z":{
        '0':tck_z[0].tolist(),
        '1':tck_z[1].tolist(),
        '2':tck_z[2]
    },
}

with open(os.path.join(script_dir, 'camera_path_spline.json'),'w+') as f:
    json.dump(res, f)

spl2 = UnivariateSpline._from_tck((res['x']['0'],res['x']['1'],res['x']['2']))

t_new = np.linspace(0, max(t), 1000)
x_new = spl2(t_new)
y_new = spline_y(t_new)
z_new = spline_z(t_new)

ax.plot(x_new, y_new, z_new, color = 'r')
print(max(t))

plt.show()