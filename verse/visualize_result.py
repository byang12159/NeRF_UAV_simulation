import pickle 
import os 
import open3d as o3d
import numpy as np 
from camera_path_spline_new import spline
from scipy.interpolate import UnivariateSpline
from drone2 import apply_model, remove_data, get_vision_estimation_batch
import matplotlib.pyplot as plt 

script_dir = os.path.dirname(os.path.realpath(__file__))

def create_box(box):
                
    translation = box[3:6]
    h, w, l = box[0], box[1], box[2]

    rotationz = box[6]
    rotationy = box[7]
    rotationx = box[8]

    # Create a bounding box outline if x,y,z is center point
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])
    
    # Create a bounding box outline if x,y,z is rear center point
    # bounding_box = np.array([
    #             [l,l,0,0,l,l,0,0],                      
    #             [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2],          
    #             [0,0,0,0,h,h,h,h]])                        

    rotation_matrixZ = np.array([
        [np.cos(rotationz), -np.sin(rotationz), 0.0],
        [np.sin(rotationz), np.cos(rotationz), 0.0],
        [0.0, 0.0, 1.0]])
    
    rotation_matrixY = np.array([
        [np.cos(rotationy), 0.0, np.sin(rotationy)],
        [0.0, 1.0 , 0.0],
        [-np.sin(rotationy), 0.0,  np.cos(rotationy)]])

    rotation_matrixX = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(rotationx), -np.sin(rotationx)],
        [0.0, np.sin(rotationx),  np.cos(rotationx)]])
    
    
    rotation_matrix = rotation_matrixZ@rotation_matrixY@rotation_matrixX
    # print(rotation_matrix)
    # Repeat the center position [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = rotation_matrix@bounding_box + eight_points.transpose()
    return corner_box.transpose()


if __name__ == "__main__":
    # Visualize environment and reachable set 
    with open(os.path.join(script_dir, 'exp2_safe.pickle'), 'rb') as f: 
        M, E, C_list, reachtube = pickle.load(f)

    # Load the .ply file
    pc_fn = os.path.join(script_dir, 'point_cloud.ply')
    point_cloud = o3d.io.read_point_cloud(pc_fn)
    # Create a red point at the center of the coordinate system
    center_point = np.array([[0.0, 0.0, 0.0]]) # Center point coordinates
    center_color = np.array([[1.0, 0.0, 0.0]]) # Red color (R,G,B)
    point1 = np.array([[0.46990477000436776,-0.01279251651773311,0.02605140075172408]])
    point2 = np.array([[-0.14787864315441807,-0.2924556311040014,-0.017899417569988384]])
    # Create coordinate frame axes
    coordinate_axes = o3d.geometry.LineSet()
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    lines = np.array([[0, 1], [0, 2], [0, 3]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # RGB colors for X, Y, Z axes
    coordinate_axes.points = o3d.utility.Vector3dVector(vertices)
    coordinate_axes.lines = o3d.utility.Vector2iVector(lines)
    coordinate_axes.colors = o3d.utility.Vector3dVector(colors)
    # Create a point cloud for the center point
    center_cloud = o3d.geometry.PointCloud()
    center_cloud.points = o3d.utility.Vector3dVector(center_point)
    center_cloud.colors = o3d.utility.Vector3dVector(center_color)
    point1_cloud = o3d.geometry.PointCloud()
    point1_cloud.points = o3d.utility.Vector3dVector(point1)
    point1_cloud.colors = o3d.utility.Vector3dVector(np.array([[0.0, 1.0, 0.0]]))
    point2_cloud = o3d.geometry.PointCloud()
    point2_cloud.points = o3d.utility.Vector3dVector(point2)
    point2_cloud.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0]]))
    # Create a visualization window
    object_list = [point_cloud, center_cloud, point1_cloud, point2_cloud, coordinate_axes]
    for i in range(0, reachtube.shape[0], 2):
        lb = reachtube[i,[1,5,9]]
        ub = reachtube[i+1,[1,5,9]]
        l, w, h = ub-lb 
        x,y,z = (lb+ub)/2 
        box = [h+0.02,w,l,x,y,z,0,0,0]

        boxes3d_pts = create_box(box)
        boxes3d_pts = o3d.utility.Vector3dVector(boxes3d_pts)
        box = o3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts)
        box.color = [1, 0, 0]  #Box color would be red box.color = [R,G,B]
        
        object_list.append(box)

    with open('vcs_sim_exp1_safe.pickle','rb') as f:
        state_list = pickle.load(f)
    with open('vcs_estimate_exp1_safe.pickle','rb') as f:
        est_list = pickle.load(f)
    with open('vcs_init_exp1_safe.pickle','rb') as f:
        e_list = pickle.load(f)

    total_data = 0
    data_satisfy = 0
    notin0 = 0
    notin1 = 0
    notin2 = 0
    notin3 = 0
    notin4 = 0
    notin5 = 0
    # Get accuracy of perception contract
    for i in range(len(state_list)):
        for j in range(len(state_list[i])):
            gt = np.array(state_list[i][j])
            est = np.array(est_list[i][j])

            cx, rx = apply_model(M[0], gt)
            cy, ry = apply_model(M[1], gt)
            cz, rz = apply_model(M[2], gt)
            croll, rroll = apply_model(M[3], gt)
            cpitch, rpitch = apply_model(M[4], gt)
            cyaw, ryaw = apply_model(M[5], gt)
            

            total_data += 1
            if (est[0]>=cx-rx and est[0]<=cx+rx and \
                est[1]>=cy-ry and est[1]<=cy+ry and \
                est[2]>=cz-rz and est[2]<=cz+rz and \
                est[3]>=croll-rroll and est[3]<=croll+rroll and \
                est[4]>=cpitch-rpitch and est[4]<=cpitch+rpitch and \
                est[5]>=cyaw-ryaw and est[5]<=cyaw+ryaw
                ):
                data_satisfy += 1 
            else:
                if not (est[0]>=cx-rx and est[0]<=cx+rx):
                    notin0 += 1 
                if not (est[1]>=cy-ry and est[1]<=cy+ry):
                    notin1 += 1                
                if not (est[2]>=cz-rz and est[2]<=cz+rz):
                    notin2 += 1
                if not (est[3]>=croll-rroll and est[3]<=croll+rroll):
                    notin3 += 1
                if not (est[4]>=cpitch-rpitch and est[4]<=cpitch+rpitch):
                    notin4 += 1
                if not (est[5]>=cyaw-ryaw and est[5]<=cyaw+ryaw):
                    notin5 += 1
    print(data_satisfy/total_data)
    print(total_data, notin0, notin1, notin2, notin3, notin4, notin5)   

    with open(os.path.join(script_dir, 'exp2_train4.pickle'),'rb') as f:
        data = pickle.load(f)

    data_removed = remove_data(data, E)
    state_array, trace_array, e_array = data_removed

    total_data = 0
    data_satisfy = 0
    notin0 = 0
    notin1 = 0
    notin2 = 0
    notin3 = 0
    notin4 = 0
    notin5 = 0
    # Get accuracy of perception contract
    for j in range(len(state_array)):
        gt = np.array(state_array[j])
        est = np.array(trace_array[j])

        cx, rx = apply_model(M[0], gt)
        cy, ry = apply_model(M[1], gt)
        cz, rz = apply_model(M[2], gt)
        croll, rroll = apply_model(M[3], gt)
        cpitch, rpitch = apply_model(M[4], gt)
        cyaw, ryaw = apply_model(M[5], gt)
        

        total_data += 1
        if (est[0]>=cx-rx and est[0]<=cx+rx and \
            est[1]>=cy-ry and est[1]<=cy+ry and \
            est[2]>=cz-rz and est[2]<=cz+rz and \
            # est[3]>=croll-rroll and est[3]<=croll+rroll and \
            # est[4]>=cpitch-rpitch and est[4]<=cpitch+rpitch and \
            est[5]>=cyaw-ryaw and est[5]<=cyaw+ryaw
            ):
            data_satisfy += 1 
        else:
            if not (est[0]>=cx-rx and est[0]<=cx+rx):
                notin0 += 1 
            if not (est[1]>=cy-ry and est[1]<=cy+ry):
                notin1 += 1                
            if not (est[2]>=cz-rz and est[2]<=cz+rz):
                notin2 += 1
            # if not (est[3]>=croll-rroll and est[3]<=croll+rroll):
            #     notin3 += 1
            # if not (est[4]>=cpitch-rpitch and est[4]<=cpitch+rpitch):
            #     notin4 += 1
            if not (est[5]>=cyaw-ryaw and est[5]<=cyaw+ryaw):
                notin5 += 1
    print(data_satisfy/total_data)
    print(total_data, notin0, notin1, notin2, notin3, notin4, notin5)   

    # Visualize simulation trajectories
    for i in range(len(state_list)):
        if i==2:
            continue
        traj = np.array(state_list[i])
        trajectory_points=  traj[:,:3]

        # 2. Create a LineSet object.
        trajectory = o3d.geometry.LineSet()

        # 3. Add the trajectory points to the LineSet as vertices.
        trajectory.points = o3d.utility.Vector3dVector(trajectory_points)

        # 4. Define the line segments by specifying pairs of indices.
        lines = [[i, i + 1] for i in range(len(trajectory_points) - 1)]
        trajectory.lines = o3d.utility.Vector2iVector(lines)

        # Optionally, set the color for the lines.
        trajectory.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(lines))])  # red color

        object_list.append(trajectory)

    tks_x = (spline['x']['0'],spline['x']['1'],spline['x']['2'])
    tks_y = (spline['y']['0'],spline['y']['1'],spline['y']['2'])
    tks_z = (spline['z']['0'],spline['z']['1'],spline['z']['2'])

    spline_x = UnivariateSpline._from_tck(tks_x)
    spline_y = UnivariateSpline._from_tck(tks_y)
    spline_z = UnivariateSpline._from_tck(tks_z)

    t = np.linspace(0,30,300)
    ref_x = spline_x(t)
    ref_y = spline_y(t)
    ref_z = spline_z(t)
    trajectory_points = np.column_stack([ref_x, ref_y, ref_z])

    # 2. Create a LineSet object.
    trajectory = o3d.geometry.LineSet()

    # 3. Add the trajectory points to the LineSet as vertices.
    trajectory.points = o3d.utility.Vector3dVector(trajectory_points)

    # 4. Define the line segments by specifying pairs of indices.
    lines = [[i, i + 1] for i in range(len(trajectory_points) - 1)]
    trajectory.lines = o3d.utility.Vector2iVector(lines)

    # Optionally, set the color for the lines.
    trajectory.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(lines))])  # red color

    object_list.append(trajectory)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for obj in object_list:
        vis.add_geometry(obj)
    # vis.add_geometry(point_cloud2)
    vis.get_render_option().line_width = 20
    # vis.get_render_option().point_size = 20
    vis.run()

    with open(os.path.join(script_dir, './exp2_train4.pickle'), 'rb') as f:
        data = pickle.load(f)
    state_array, trace_array, E_array = data

    miss_array = np.zeros(len(E))
    total_array = np.zeros(len(E))
    for i, Ep in enumerate(E):
        in_part = np.where(
            (Ep[0,0]<=E_array[:,0]) & \
            (E_array[:,0]<=Ep[1,0]) & \
            (Ep[0,1]<=E_array[:,1]) & \
            (E_array[:,1]<=Ep[1,1])
        )[0]
        total_array[i] = len(in_part)
        state_contain = state_array[in_part]
        trace_contain = trace_array[in_part]
        E_contain = E_array[in_part]
        # for j in range(len(in_part)):
        for j in range(len(state_contain)):
            lb, ub = get_vision_estimation_batch(state_contain[j,:], M)
            if trace_contain[j,0]<lb[0] or trace_contain[j,0]>ub[0] or \
                trace_contain[j,1]<lb[1] or trace_contain[j,1]>ub[1] or \
                trace_contain[j,2]<lb[2] or trace_contain[j,2]>ub[2] or \
                trace_contain[j,5]<lb[3] or trace_contain[j,5]>ub[3]:
                pass
                miss_array[i] += 1

    accuracy_array = (total_array-miss_array)/total_array
    print((total_array-miss_array).sum()/total_array.sum())



    full_e = np.ones((10, 10))*(-1)
    min_acc = float('inf')
    min_acc_e1 = 0
    min_acc_e2 = 0
    for i in range(len(E)):
        E_part = E[i]
        idx1 = round((E_part[0,0]-0.0)/0.1)
        idx2 = round((E_part[0,1]-(-1.0))/0.1)
        full_e[idx1, idx2] = accuracy_array[i]
        if accuracy_array[i]!=0 and accuracy_array[i]<min_acc:
            min_acc = accuracy_array[i]
            min_acc_e1 = E_part[0,0]
            min_acc_e2 = E_part[0,1]
    print(min_acc, min_acc_e1, min_acc_e2)
    # full_e = full_e/np.max(full_e)
    rgba_image = np.zeros((10, 10, 4))  # 4 channels: R, G, B, A
    rgba_image[..., :3] = plt.cm.viridis(full_e)[..., :3]  # Apply a colormap    
    mask = np.where(full_e<0)
    rgba_image[..., 3] = 1.0  # Set alpha to 1 (non-transparent)

    # Apply the mask to make some pixels transparent
    rgba_image[mask[0], mask[1], :3] = 1  # Set alpha to 0 (transparent) for masked pixels
    plt.imshow(rgba_image)

    # o3d.visualization.draw_geometries(object_list, window_name="Point Cloud with Axes", width=800, height=600)
    
    # ax = plt.gca()
    # ax.set_xticks(np.round(np.arange(0,12,2)-0.5,1))
    # ax.set_yticks(np.round(np.arange(0,12,2)-0.5,1))
    # ax.set_xticklabels(np.round(np.arange(-1.0, 0.05, 0.2),2), fontsize=14)
    # ax.set_yticklabels(np.round(np.arange(0, 1.05, 0.2),2), fontsize=14)
    # plt.xlabel('Ambient light intensity', fontsize=16)
    # plt.ylabel('Fog level', fontsize=16)
    # # plt.xticks(list(range(0,14,2)), np.round(np.arange(-0.05,0.6,0.1),2))
    # # plt.yticks(list(range(0,20,2)), np.round(np.arange(0.25, 1.2, 0.1),2))


    # plt.show()
