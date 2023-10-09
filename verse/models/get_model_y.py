import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import os 
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, QuantileRegressor
import statsmodels.api as sm 
import json 

def compute_model_y(state_array, trace_array, E_array, pc=0.9, pr=0.95):
    X = state_array[:,0:3]
    Y = trace_array[:,1]
    model_radius = sm.QuantReg(Y, sm.add_constant(X))
    result = model_radius.fit(q = 0.5)
    cc = result.params 

    # Getting Model for Radius
    X = state_array[:,0:3]
    X_radius = cc[0] + cc[1]*X[:,0] + cc[2]*X[:,1] + cc[3]*X[:,2]
    trace_array_radius = trace_array[:,1]

    Y_radius = np.abs(trace_array_radius-X_radius)
    # Y_radius = np.abs(trace_list_radius[:,0]-X_radius)
    quantile = pr
    model_radius = sm.QuantReg(Y_radius, sm.add_constant(X))
    result = model_radius.fit(q=quantile)
    cr = result.params

    res = {
        'dim': 'y',
        'coef_center':cc.tolist(),
        'coef_radius': cr.tolist()
    }
    return res
# -------------------------------------

# Testing the obtained models
# The center of perception contract. 
# mcc # Input to this function is the ground truth state and center of range of environmental parameter

# # The radius of possible center 
# model_center_radius 
# ccr # Input to this function is the ground truth state and center of range of environmental parameter

# # The radius of perception contract 
# model_radius 
# cr # Input to this function is the ground truth state

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, '../exp2_train1.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)

    state_array, trace_array, E_array = data
    res = compute_model_y(state_array, trace_array, E_array, pc = 0.8, pr=0.999)
    cc = res['coef_center']
    cr = res['coef_radius']
    with open(os.path.join(script_dir,'model_y2.json'),'w+') as f:
        json.dump(res, f, indent=4)

    sample_contained = 0
    total_sample = 0
    for i in range(state_array.shape[0]):
        x = state_array[i,0]
        y = state_array[i,1]
        z = state_array[i,2]
        center_center = cc[0] + cc[1]*x + cc[2]*y + cc[3]*z
        radius = cr[0] + x*cr[1] + y*cr[2] + z*cr[3]
        x_est = trace_array[i, 1]
        if x_est<center_center+radius and \
            x_est>center_center-radius:
            sample_contained += 1
            total_sample += 1 
        else:
            total_sample += 1

    print(sample_contained/total_sample)
    plt.plot(state_array[:,1], trace_array[:,1], 'b*')
    c = cc[0] + cc[1]*state_array[:,0] + cc[2]*state_array[:,1] + cc[3]*state_array[:,2]
    r = cr[0] + cr[1]*state_array[:,0] + cr[2]*state_array[:,1] + cr[3]*state_array[:,2]
    plt.plot(state_array[:,1], c-r, 'r*')
    plt.plot(state_array[:,1], c+r, 'r*')
    
    plt.show()