"""
INPUT:  dataset.log files
OUTPUT1-> log_info: 1D list, showing basic parameters corresponding to each run
OUTPUT2-> data: 1D list, each entry is a 2D np.array of shape (cycle_points x 14). 6 columns for gt_state, 6 columns for est_state, 2 columns for [param_fog, param_dark]
"""
import numpy as np
import re
import os 
import pickle 
import copy

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
state_array1 = None 
trace_array1 = None
env_array1 = None 
for i in range(len(data)):
    for j in range(len(data[i])):   
        state = data[i][j][:6]
        trace = data[i][j][6:12]
        env = data[i][j][12:]
        if state_array1 is None:
            state_array1 = state.reshape((1,-1)) 
        else: 
            state_array1 = np.vstack((state_array1, copy.deepcopy(state.reshape((1, -1)))))

        if trace_array1 is None:
            trace_array1 = trace.reshape((1,-1)) 
        else: 
            trace_array1 = np.vstack((trace_array1, copy.deepcopy(trace.reshape((1, -1)))))

        if env_array1 is None:
            env_array1 = env.reshape((1,-1)) 
        else: 
            env_array1 = np.vstack((env_array1, copy.deepcopy(env.reshape((1, -1)))))

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
state_array2 = None 
trace_array2 = None
env_array2 = None 
for i in range(len(data)):
    for j in range(len(data[i])):   
        state = data[i][j][:6]
        trace = data[i][j][6:12]
        env = data[i][j][12:]
        if state_array2 is None:
            state_array2 = state.reshape((1,-1)) 
        else: 
            state_array2 = np.vstack((state_array2, copy.deepcopy(state.reshape((1, -1)))))

        if trace_array2 is None:
            trace_array2 = trace.reshape((1,-1)) 
        else: 
            trace_array2 = np.vstack((trace_array2, copy.deepcopy(trace.reshape((1, -1)))))

        if env_array2 is None:
            env_array2 = env.reshape((1,-1)) 
        else: 
            env_array2 = np.vstack((env_array2, copy.deepcopy(env.reshape((1, -1)))))

state_array = np.vstack((state_array1, state_array2))
trace_array = np.vstack((trace_array1, trace_array2))
env_array = np.vstack((env_array1, env_array2))
with open(os.path.join(script_dir, 'exp2_train3.pickle'), 'wb+') as f:
    pickle.dump((state_array2, trace_array2, env_array2), f)