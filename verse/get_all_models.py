from models.get_model_x import compute_model_x
from models.get_model_y import compute_model_y
from models.get_model_z import compute_model_z

import os 
import pickle 
import numpy as np
import json 

def get_all_models(data):
    state_array, trace_array, E_array = data

    model_x = compute_model_x(state_array, trace_array, E_array, 0.5, 0.94)
    model_y = compute_model_y(state_array, trace_array, E_array, 0.5, 0.94)
    model_z = compute_model_z(state_array, trace_array, E_array, 0.5, 0.94)

    return model_x, model_y, model_z

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