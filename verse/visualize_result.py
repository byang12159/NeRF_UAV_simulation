import pickle 
import os 
import open3d as o3d
import numpy as np 
from camera_path_spline import spline
from scipy.interpolate import UnivariateSpline
from drone2 import apply_model

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
        h, w, l = ub-lb 
        x,y,z = (lb+ub)/2 
        box = [h,w,l,x,y,z,0,0,0]

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
    # Get accuracy of perception contract
    for i in range(len(state_list)):
        for j in range(len(state_list[i])):
            gt = np.array(state_list[i][j])
            est = np.array(est_list[i][j])

            cx, rx = apply_model(M[0], gt)
            cy, ry = apply_model(M[1], gt)
            cz, rz = apply_model(M[2], gt)

            total_data += 1
            if (est[0]>=cx-rx and est[0]<=cx+rx and \
                est[1]>=cy-ry and est[1]<=cy+ry and \
                est[2]>=cz-rz and est[2]<=cz+rz):
                data_satisfy += 1 
            else:
                if not (est[0]>=cx-rx and est[0]<=cx+rx):
                    notin0 += 1 
                if not (est[1]>=cy-ry and est[1]<=cy+ry):
                    notin1 += 1                
                if not (est[2]>=cz-rz and est[2]<=cz+rz):
                    notin2 += 1
    print(data_satisfy/total_data)
    print(total_data, notin0, notin1, notin2)
    

    # Visualize simulation trajectories
    for i in range(len(state_list)):
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

    o3d.visualization.draw_geometries(object_list, window_name="Point Cloud with Axes", width=800, height=600)
    
