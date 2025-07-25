import os
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import trimesh
import tkinter as tk
from tkinter import filedialog

def ensure_xyz_header(csv_path):
    with open(csv_path, "r") as f:
        first_line = f.readline().strip().lower()
        if all(h in first_line for h in ['x', 'y', 'z']):
            return
    with open(csv_path, "r") as infile:
        lines = infile.readlines()
    with open(csv_path, "w") as outfile:
        outfile.write("x,y,z\n")
        outfile.writelines(lines)

def parse_csv(path):
    ensure_xyz_header(path)
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    xyz = [cols.index('x'), cols.index('y'), cols.index('z')]
    df = df.iloc[:, xyz]
    df.columns = ['x', 'y', 'z']
    return df

def parse_obj(path):
    mesh = trimesh.load(path, force='mesh')
    verts = mesh.vertices
    if verts.shape[0] == 0:
        cloud = trimesh.load(path, force='point')
        verts = cloud.vertices
    return pd.DataFrame(verts, columns=['x', 'y', 'z'])

def parse_pts(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.strip() and not line[0].isalpha():
                parts = line.strip().split()
                if len(parts) >= 3:
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return pd.DataFrame(data, columns=['x', 'y', 'z'])

def parse_ply(path):
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("Install plyfile: pip install plyfile")
    ply = PlyData.read(path)
    v = ply['vertex']
    return pd.DataFrame({'x': v['x'], 'y': v['y'], 'z': v['z']})

def convert_txt_to_csv(txt_path):
    with open(txt_path, "r") as infile:
        first_line = infile.readline()
        has_comma = ',' in first_line
    csv_path = os.path.splitext(txt_path)[0] + ".csv"
    if has_comma:
        with open(txt_path, "r") as infile, open(csv_path, "w") as outfile:
            for line in infile:
                outfile.write(line)
        ensure_xyz_header(csv_path)
        return csv_path
    with open(txt_path, "r") as infile, open(csv_path, "w") as outfile:
        lines = infile.readlines()
        first_line = lines[0].strip().lower()
        if all(h in first_line for h in ['x', 'y', 'z']):
            outfile.write(",".join(first_line.split()) + "\n")
            data_lines = lines[1:]
        else:
            outfile.write("x,y,z\n")
            data_lines = lines
        for line in data_lines:
            line = line.strip()
            if line:
                parts = line.split()
                outfile.write(",".join(parts) + "\n")
    return csv_path

def load_xyz(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return parse_csv(path)
    elif ext == '.obj':
        return parse_obj(path)
    elif ext == '.pts':
        return parse_pts(path)
    elif ext == '.ply':
        return parse_ply(path)
    elif ext == '.txt':
        csv_path = convert_txt_to_csv(path)
        return parse_csv(csv_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def relative_to_first(df, decimals=2):
    ref = df.iloc[0][['x', 'y']].values.astype(float)
    df[['x', 'y']] = np.round(df[['x', 'y']].astype(float) - ref, decimals)
    print(f"Shifted X and Y so first row is at (0,0): ({ref[0]}, {ref[1]}) subtracted from all X and Y.")
    return df

def mesh_from_points(
    df,
    output_folder,
    mesh_name="mesh.obj",
    max_edge_length=5.0,
    axis_proj='XY'
):
    verts = df[['x', 'y', 'z']].values
    if verts.shape[0] == 0:
        print("No vertices found!")
        return None

    # Project to 2D for triangulation
    if axis_proj == 'XY':
        points2d = verts[:, :2]
    elif axis_proj == 'XZ':
        points2d = verts[:, [0,2]]
    elif axis_proj == 'YZ':
        points2d = verts[:, 1:]
    else:
        raise ValueError("axis_proj must be 'XY', 'XZ', or 'YZ'")

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

    if not faces:
        print("No faces created. Try increasing max_edge_length.")
        return None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    obj_path = os.path.join(output_folder, mesh_name)
    mesh.export(obj_path)
    print(f"Exported mesh OBJ to {obj_path}")
    return obj_path

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select point cloud file",
        filetypes=[("Point cloud files", ".csv .obj .pts .ply .txt"), ("All files", ".*")]
    )
    root.destroy()
    if not file_path:
        print("No file selected. Exiting.")
        return

    df = load_xyz(file_path)
    print(f"Loaded {len(df)} points from {file_path}")

    # Shift X and Y so first row is at (0,0) and print info
    df = relative_to_first(df, decimals=2)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = os.path.join(os.path.dirname(file_path), base_name + "_mesh")
    os.makedirs(output_folder, exist_ok=True)

    mesh_name = base_name + "_delaunay.obj"
    mesh_from_points(
        df,
        output_folder,
        mesh_name=mesh_name,
        max_edge_length=5.0,   # You can change this value
        axis_proj='XY'         # Or 'XZ', 'YZ' as needed
    )

if __name__ == "__main__":
    main()