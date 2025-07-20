import numpy as np
from scipy.spatial import Delaunay
import trimesh
import os
import tkinter as tk
from tkinter import filedialog

def delaunay_faces_with_max_edge(verts, max_edge_length=1.0, axis_proj='XY'):
    if verts.shape[0] == 0:
        print("No vertices found!")
        return []
    # Project to 2D
    if axis_proj == 'XY':
        points2d = verts[:, :2]
    elif axis_proj == 'XZ':
        points2d = verts[:, [0,2]]
    elif axis_proj == 'YZ':
        points2d = verts[:, 1:]
    else:
        raise ValueError("axis_proj must be 'XY', 'XZ', or 'YZ'")
    # Delaunay triangulation
    tri = Delaunay(points2d)
    faces = []
    for simplex in tri.simplices:
        a, b, c = simplex
        pa, pb, pc = verts[a], verts[b], verts[c]
        ab = np.linalg.norm(pa - pb)
        bc = np.linalg.norm(pb - pc)
        ca = np.linalg.norm(pc - pa)
        if ab <= max_edge_length and bc <= max_edge_length and ca <= max_edge_length:
            faces.append([a, b, c])
    return faces

# File dialog
root = tk.Tk()
root.withdraw()
input_path = filedialog.askopenfilename(
    title="Select OBJ file",
    filetypes=[("OBJ files", "*.obj")]
)
root.destroy()

if not input_path:
    print("No file selected. Exiting.")
    exit()

# Try loading as mesh
mesh = trimesh.load(input_path, force='mesh')
verts = mesh.vertices
print("Loaded vertices shape (mesh):", verts.shape)

# If empty, try as point cloud
if verts.shape[0] == 0:
    cloud = trimesh.load(input_path, force='point')
    verts = cloud.vertices
    print("Loaded vertices shape (point):", verts.shape)

if verts.shape[0] == 0:
    print("No vertices found in OBJ file! Exiting.")
    exit()

# Delaunay triangulation
max_edge_length = 5  # Adjust as needed
axis_proj = 'XY'       # 'XY', 'XZ', or 'YZ'
faces = delaunay_faces_with_max_edge(verts, max_edge_length, axis_proj)

if not faces:
    print("No faces created. Try increasing max_edge_length.")
    exit()

# Create new mesh and export
new_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
base, ext = os.path.splitext(input_path)
output_path = base + "_delaunay.obj"
new_mesh.export(output_path)
print(f"Exported triangulated OBJ to {output_path}")