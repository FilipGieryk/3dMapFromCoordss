import csv
import trimesh
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

# --- Open file dialog to select CSV ---
root = tk.Tk()
root.withdraw()  # Hide the main window
csv_path = filedialog.askopenfilename(
    title="Select CSV file",
    filetypes=[("CSV files", "*.csv")]
)
root.destroy()

if not csv_path:
    print("No file selected. Exiting.")
    exit()

# --- Output OBJ will be in the same folder as the script ---
script_dir = os.path.dirname(os.path.realpath(__file__))
export_path = os.path.join(script_dir, "currentdata.obj")

# --- Read CSV ---
points = []
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        points.append([float(row['x']), float(row['y']), float(row['z'])])

if not points:
    print("No points found in CSV!")
    exit()

points = np.array(points)

# --- Create a point cloud mesh (vertices only, no faces) ---
cloud = trimesh.points.PointCloud(points)

# --- Export as OBJ ---
cloud.export(export_path)
print(f"Exported OBJ to {export_path}")