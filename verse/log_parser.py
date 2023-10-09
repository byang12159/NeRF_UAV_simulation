"""
INPUT:  dataset.log files
OUTPUT1-> log_info: 1D list, showing basic parameters corresponding to each run
OUTPUT2-> data: 1D list, each entry is a 2D np.array of shape (cycle_points x 14). 6 columns for gt_state, 6 columns for est_state, 2 columns for [param_fog, param_dark]
"""
import numpy as np
import re
import os 
import pickle 

# Open the log file for reading
script_dir = os.path.dirname(os.path.realpath(__file__))
log_file_path = os.path.join(script_dir, 'dataset.log')  # Replace with your log file path
cycle_points = 80

# Initialize 
log_info = []
data = []
tracking = -1

def extract(input_string):

    tuple_pattern = r'\((.*?)\)'  # Pattern to match everything inside parentheses
    tuples = re.findall(tuple_pattern, input_string)
    output_string = tuples[0].replace('[', '').replace(']', '')
    numbers = np.fromstring(output_string, dtype = float, sep = ',') 

    return numbers

# try:
with open(log_file_path, 'r') as log_file:
    for index, line in enumerate(log_file):
        if line.startswith("Env"):
            log_info.append(line)
            tracking = 0
            cycle_data = np.zeros((cycle_points,14))
            continue

        if tracking != -1:
            cycle_data[tracking] = extract(line)
            tracking += 1
            
        
        if tracking == cycle_points:
            data.append(cycle_data)
            print(cycle_data)
            print("cycledata",cycle_data.shape)
            tracking = -1

# except FileNotFoundError:
#     print(f"Log file '{log_file_path}' not found.")
# except Exception as e:
#     print(f"An error occurred: {e}")


print(log_info)
print(len(data))
print(data[0][0])
state_array = None 
trace_array = None
env_array = None 
for i in range(len(data)):
    for j in range(len(data[i])):   
        state = data[i][j][:6]
        trace = data[i][j][6:12]
        env = data[i][j][12:]
        if state_array is None:
            state_array = state.reshape((1,-1)) 
        else: 
            state_array = np.vstack((state_array, state.reshape((1, -1))))

        if trace_array is None:
            trace_array = trace.reshape((1,-1)) 
        else: 
            trace_array = np.vstack((trace_array, trace.reshape((1, -1))))

        if env_array is None:
            env_array = env.reshape((1,-1)) 
        else: 
            env_array = np.vstack((env_array, env.reshape((1, -1))))

with open(os.path.join(script_dir, 'exp1_train1.pickle'), 'wb+') as f:
    pickle.dump((state_array, trace_array, env_array), f)