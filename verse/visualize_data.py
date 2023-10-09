import pickle
import matplotlib.pyplot as plt 
import os 

script_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(script_dir, 'exp2_train1.pickle'), 'rb') as f:
    data = pickle.load(f)

state_array, trace_array, env_array = data 

plt.figure(0)
plt.plot(state_array[:,0], trace_array[:,0], '*')
plt.title('x')

plt.figure(1)
plt.plot(state_array[:,1], trace_array[:,1], '*')
plt.title('y')

plt.figure(2)
plt.plot(state_array[:,2], trace_array[:,2], '*')
plt.title('z')

plt.figure(3)
plt.plot(state_array[:,3], trace_array[:,3], '*')
plt.title('roll')

plt.figure(4)
plt.plot(state_array[:,4], trace_array[:,4], '*')
plt.title('pitch')

plt.figure(5)
plt.plot(state_array[:,5], trace_array[:,5], '*')
plt.title('yaw')

plt.show()