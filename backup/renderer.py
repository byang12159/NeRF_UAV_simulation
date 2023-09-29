import open3d as o3d
import numpy as np

# Load the .ply file
point_cloud = o3d.io.read_point_cloud("point_cloud.ply")

# Create a red point at the center of the coordinate system
center_point = np.array([[0.0, 0.0, 0.0]])  # Center point coordinates
center_color = np.array([[1.0, 0.0, 0.0]])  # Red color (R,G,B)

# Create coordinate frame axes
coordinate_axes = o3d.geometry.LineSet()
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
lines = np.array([[0, 1], [0, 2], [0, 3]])
colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # RGB colors for X, Y, Z axes
coordinate_axes.points = o3d.utility.Vector3dVector(vertices)
coordinate_axes.lines = o3d.utility.Vector2iVector(lines)
coordinate_axes.colors = o3d.utility.Vector3dVector(colors)

# Create a point cloud for the center point
center_cloud = o3d.geometry.PointCloud()
center_cloud.points = o3d.utility.Vector3dVector(center_point)
center_cloud.colors = o3d.utility.Vector3dVector(center_color)

# Create a visualization window
o3d.visualization.draw_geometries([point_cloud, center_cloud, coordinate_axes], window_name="Point Cloud with Axes", width=800, height=600)



