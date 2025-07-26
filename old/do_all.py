import pandas as pd
import numpy as np
from scipy.spatial import cKDTree, Delaunay
import trimesh
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import datetime

def downsample_points(points, min_distance):
    tree = cKDTree(points)
    mask = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if not mask[i]:
            continue
        idxs = tree.query_ball_point(point, min_distance)
        idxs = [idx for idx in idxs if idx > i]
        mask[idxs] = False
    return mask

def downsample_csv(df, min_distance):
    points = df[['x', 'y', 'z']].values
    mask = downsample_points(points, min_distance)
    return df[mask]

def normalize_df(df):
    x0, y0, z0 = df.iloc[0][['x', 'y', 'z']]
    df['x'] = (df['x'] - x0).round(6)
    df['y'] = (df['y'] - y0).round(6)
    df['z'] = (df['z'] - z0).round(6)
    return df

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

class DownsampleDelaunayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV → Downsample/Normalize → OBJ → Delaunay")
        self.df = None
        self.filename = None

        self.file_label = tk.Label(root, text="No file selected")
        self.file_label.pack(pady=5)

        self.row_label = tk.Label(root, text="")
        self.row_label.pack(pady=5)

        self.slider = tk.Scale(root, from_=0.001, to=1.0, resolution=0.001,
                               orient=tk.HORIZONTAL, label="Min Distance", length=300)
        self.slider.set(0.05)
        self.slider.pack(pady=5)

        self.edge_label = tk.Label(root, text="Max Delaunay Edge Length")
        self.edge_label.pack(pady=2)
        self.edge_entry = tk.Entry(root)
        self.edge_entry.insert(0, "0.5")
        self.edge_entry.pack(pady=2)

        self.proj_label = tk.Label(root, text="Projection (XY, XZ, YZ)")
        self.proj_label.pack(pady=2)
        self.proj_var = tk.StringVar(value="XY")
        self.proj_menu = tk.OptionMenu(root, self.proj_var, "XY", "XZ", "YZ")
        self.proj_menu.pack(pady=2)

        self.open_button = tk.Button(root, text="Open CSV", command=self.open_file)
        self.open_button.pack(pady=5)

        self.apply_button = tk.Button(root, text="Process and Export", command=self.apply, state=tk.DISABLED)
        self.apply_button.pack(pady=10)

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Select input CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
            if not set(['x', 'y', 'z']).issubset(df.columns):
                raise ValueError("CSV must have columns: x, y, z")
            self.df = df
            self.filename = os.path.basename(file_path)
            self.file_label.config(text=f"File: {self.filename}")
            self.row_label.config(text=f"Rows: {len(df)}")
            self.apply_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.df = None
            self.filename = None
            self.file_label.config(text="No file selected")
            self.row_label.config(text="")
            self.apply_button.config(state=tk.DISABLED)

    def apply(self):
        if self.df is None:
            return
        min_distance = self.slider.get()
        try:
            # Downsample
            df_down = downsample_csv(self.df, min_distance)
            # Normalize
            df_norm = normalize_df(df_down)
            points = df_norm[['x', 'y', 'z']].values

            # Save point cloud OBJ
            script_dir = os.path.dirname(os.path.realpath(__file__))
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            obj_path = os.path.join(script_dir, f"downsampled_{now}.obj")
            cloud = trimesh.points.PointCloud(points)
            cloud.export(obj_path)

            # Delaunay triangulation
            try:
                max_edge_length = float(self.edge_entry.get())
            except:
                max_edge_length = 0.5
            axis_proj = self.proj_var.get()
            faces = delaunay_faces_with_max_edge(points, max_edge_length, axis_proj)
            if not faces:
                messagebox.showwarning("Warning", "No faces created. Try increasing max edge length.")
                return

            # Save triangulated OBJ
            tri_obj_path = os.path.join(script_dir, f"downsampled_{now}_delaunay.obj")
            new_mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
            new_mesh.export(tri_obj_path)

            messagebox.showinfo("Success", f"Saved:\n{os.path.basename(obj_path)}\n{os.path.basename(tri_obj_path)}\nRows: {len(df_norm)}\nFaces: {len(faces)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    root = tk.Tk()
    app = DownsampleDelaunayApp(root)
    root.mainloop()