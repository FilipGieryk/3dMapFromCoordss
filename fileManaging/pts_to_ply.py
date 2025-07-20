import numpy as np
import open3d as o3d

def load_pts(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    try:
        float(lines[0].strip().split()[0])
        start = 0
    except ValueError:
        start = 1

    points = []
    for line in lines[start:]:
        coords = list(map(float, line.strip().split()))
        if coords:
            points.append(coords)
    
    return np.array(points)

# Load .pts file
points = load_pts("your_file.pts")

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Save to PLY
o3d.io.write_point_cloud("converted.ply", pcd)
