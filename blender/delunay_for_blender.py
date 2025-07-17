import bpy
import numpy as np
from scipy.spatial import Delaunay

def delaunay_faces_with_max_edge(obj, max_edge_length=1.0, axis_proj='XY'):
    """
    Given a mesh object with only vertices, create faces using Delaunay triangulation,
    but only keep triangles where all edges are <= max_edge_length.
    axis_proj: 'XY', 'XZ', or 'YZ' - which 2D plane to project to.
    """
    mesh = obj.data
    verts = np.array([v.co for v in mesh.vertices])
    
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
        # Compute edge lengths
        ab = np.linalg.norm(pa - pb)
        bc = np.linalg.norm(pb - pc)
        ca = np.linalg.norm(pc - pa)
        # Only keep triangle if all edges are short enough
        if ab <= max_edge_length and bc <= max_edge_length and ca <= max_edge_length:
            faces.append([a, b, c])
    
    # Assign faces to mesh
    mesh.clear_geometry()
    mesh.from_pydata(verts.tolist(), [], faces)
    mesh.update()
    print(f"Created {len(faces)} faces from {len(verts)} vertices (max edge: {max_edge_length}).")

# Usage:
# Select your mesh object in the viewport
obj = bpy.context.active_object
delaunay_faces_with_max_edge(obj, max_edge_length=2.0, axis_proj='XY')  # Adjust max_edge_length as needed