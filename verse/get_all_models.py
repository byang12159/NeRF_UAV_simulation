from models.get_model_x import compute_model_x
from models.get_model_y import compute_model_y
from models.get_model_z import compute_model_z
from models.get_model_yaw import compute_model_yaw
from models.get_model_pitch import compute_model_pitch
from models.get_model_roll import compute_model_roll

import os 
import pickle 
import numpy as np
import json 

def get_all_models(data):
    state_array, trace_array, E_array = data

    model_x = compute_model_x(state_array, trace_array, E_array, 0.5, 0.96)
    model_y = compute_model_y(state_array, trace_array, E_array, 0.5, 0.96)
    model_z = compute_model_z(state_array, trace_array, E_array, 0.5, 0.96)
    model_roll = compute_model_roll(state_array, trace_array, E_array, 0.5, 0.96)
    model_pitch = compute_model_pitch(state_array, trace_array, E_array, 0.5, 0.96) 
    model_yaw = compute_model_yaw(state_array, trace_array, E_array, 0.5, 0.96)
    # model_x = {
    #     'dim': 'x',
    #     'coef_center':[0,1,0,0],
    #     'coef_radius': [0.05,0,0,0]
    # }
    # model_y = {
    #     'dim': 'y',
    #     'coef_center':[0,0,1,0],
    #     'coef_radius': [0.05,0,0,0]
    # }
    # model_z = {
    #     'dim': 'z',
    #     'coef_center': [0,0,0,1],
    #     'coef_radius': [0.05,0,0,0]
    # }
    # model_roll = {
    #     'dim': 'roll',
    #     'coef_center': [-0.05,0,0,0,1],
    #     # 'coef_center': [0,0,0,0,1],
    #     'coef_radius': [0.01,0,0,0,0]
    # }
    # model_pitch = {
    #     'dim': 'pitch',
    #     'coef_center': [-0.04,0,0,0,1],
    #     # 'coef_center': [0,0,0,0,1],
    #     'coef_radius': [0.01,0,0,0,0]
    # }
    # model_yaw = {
    #     'dim': 'yaw',
    #     'coef_center': [-0.05,0,0,0,1],
    #     # 'coef_center': [0,0,0,0,1],
    #     'coef_radius': [0.01,0,0,0,0]
    # }
    
    return model_x, model_y, model_z, model_roll, model_pitch, model_yaw

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(script_dir, './exp2_train1.pickle'), 'rb') as f:
        data = pickle.load(f)

    model_x, model_y, model_z = get_all_models(data)
    with open(os.path.join(script_dir, './models/model_x2.json'), 'w+') as f:
        json.dump(model_x, f)

    with open(os.path.join(script_dir, './models/model_y2.json'), 'w+') as f:
        json.dump(model_y, f)

    with open(os.path.join(script_dir, './models/model_z2.json'), 'w+') as f:
        json.dump(model_z, f)