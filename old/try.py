import open3d as o3d

def ply_to_mesh(input_ply, output_mesh_obj, poisson_depth=11):
    # 1. Load point cloud from PLY
    pcd = o3d.io.read_point_cloud(input_ply)
    if len(pcd.points) == 0:
        raise ValueError("Point cloud not loaded or is empty. Check your PLY file.")

    # 2. Remove outliers (optional, helps with noise)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 3. Estimate normals (required for Poisson)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    # 4. Poisson surface reconstruction (high detail)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)

    # 5. Save mesh
    o3d.io.write_triangle_mesh(output_mesh_obj, mesh)
    print(f"Mesh saved to {output_mesh_obj}")

# Example usage:
ply_to_mesh("car.ply", "car_mesh.obj", poisson_depth=11)