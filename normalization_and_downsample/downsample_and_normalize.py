import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
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

def estimate_downsample(df, min_distance, sample_size=1000):
    if len(df) <= sample_size:
        mask = downsample_points(df[['x', 'y', 'z']].values, min_distance)
        return mask.sum()
    else:
        sample = df.sample(sample_size, random_state=42)
        mask = downsample_points(sample[['x', 'y', 'z']].values, min_distance)
        ratio = mask.sum() / sample_size
        return int(len(df) * ratio)

class DownsampleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Downsample & Normalize CSV")
        self.df = None
        self.filename = None

        self.file_label = tk.Label(root, text="No file selected")
        self.file_label.pack(pady=5)

        self.row_label = tk.Label(root, text="")
        self.row_label.pack(pady=5)

        self.slider = tk.Scale(root, from_=0.001, to=1.0, resolution=0.001,
                               orient=tk.HORIZONTAL, label="Min Distance", length=300,
                               command=self.update_estimate)
        self.slider.set(0.05)
        self.slider.pack(pady=5)

        self.estimate_label = tk.Label(root, text="Estimated rows after downsample: -")
        self.estimate_label.pack(pady=5)

        self.open_button = tk.Button(root, text="Open CSV", command=self.open_file)
        self.open_button.pack(pady=5)

        self.apply_button = tk.Button(root, text="Apply Downsample & Normalize", command=self.apply, state=tk.DISABLED)
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
            self.update_estimate()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.df = None
            self.filename = None
            self.file_label.config(text="No file selected")
            self.row_label.config(text="")
            self.apply_button.config(state=tk.DISABLED)
            self.estimate_label.config(text="Estimated rows after downsample: -")

    def update_estimate(self, event=None):
        if self.df is not None:
            min_distance = self.slider.get()
            try:
                est = estimate_downsample(self.df, min_distance)
                self.estimate_label.config(
                    text=f"Estimated rows after downsample: {est}"
                )
            except Exception as e:
                self.estimate_label.config(text=f"Estimate error: {e}")

    def apply(self):
        if self.df is None:
            return
        min_distance = self.slider.get()
        try:
            # Downsample
            df_down = downsample_csv(self.df, min_distance)
            # Normalize
            df_norm = normalize_df(df_down)
            # Save
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = f"downsampled_normalized_{now}.csv"
            out_path = os.path.join(os.path.dirname(__file__), out_name)
            df_norm.to_csv(out_path, index=False)
            messagebox.showinfo("Success", f"Saved: {out_name}\nRows: {len(df_norm)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = DownsampleApp(root)
    root.mainloop()