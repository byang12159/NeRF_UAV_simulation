import json
import numpy as np
# Open the JSON file
with open('camera_path-2.json', 'r') as file:
    # Read the JSON data into a Python data structure
    data = json.load(file)

# Now you can work with the 'data' variable, which contains the JSON content
# print(data.shape)
campath = data.get('camera_path')
print(type(data.get('camera_path')))
print(len(campath))
print(campath[0])
print(campath[0].get('camera_to_world'))

print("renderheight",data.get('render_height'))
print("rendereifth",data.get('render_width'))

cams = data.get('keyframes')
print(cams[0].get('fov'))

cam_info = np.zeros((len(campath),16))
for j in range(len(campath)):
    cam_info[j] = campath[0].get('camera_to_world')

# print(values_list)
print("DONE")
