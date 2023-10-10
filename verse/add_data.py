import numpy as np 
import pickle 
import os 

with open('vcs_sim_exp1_safe.pickle','rb') as f:
    state_list = pickle.load(f)
with open('vcs_estimate_exp1_safe.pickle','rb') as f:
    est_list = pickle.load(f)
with open('vcs_init_exp1_safe.pickle','rb') as f:
    e_list = pickle.load(f)

script_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(script_dir, 'exp2_train1.pickle'),'rb') as f:
    state_array, trace_array, e_array = pickle.load(f)

len_state = len(state_array)
len_trace = len(trace_array)
len_e = len(e_array)
for i in range(len(state_list)):
    e = e_list[i]
    for j in range(len(state_list[i])):
        state = state_list[i][j]
        est = est_list[i][j]

        state_array = np.concatenate((state_array, state.reshape((1,-1))))
        trace_array = np.concatenate((trace_array, est.reshape((1,-1))))
        e_array = np.concatenate((e_array, e[1].reshape((1,-1))))

with open(os.path.join(script_dir, 'exp2_train2.pickle'), 'wb+') as f:
    pickle.dump((state_array[len_state:,:], trace_array[len_trace:,:], e_array[len_e:,:]), f)