import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
import tkinter as tk
from tkinter import filedialog
import os

# Open file dialog to select CSV
root = tk.Tk()
root.withdraw()
csv_path = filedialog.askopenfilename(
    title="Select CSV file",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)
if not csv_path:
    print("No file selected.")
    exit()

# Load CSV
df = pd.read_csv(csv_path)
print("CSV columns:", df.columns)

x = df['x'].values
y = df['y'].values
z = df['z'].values

# Grid resolution
grid_size = 4097
print(f"Grid size: {grid_size} x {grid_size} (nearest-neighbor fill)")

# Grid bounds
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

x_res = (xmax - xmin) / (grid_size - 1)
y_res = (ymax - ymin) / (grid_size - 1)

# Create target grid
grid_x, grid_y = np.meshgrid(
    np.linspace(xmin, xmax, grid_size),
    np.linspace(ymin, ymax, grid_size)
)

# Flatten grid and find nearest CSV point for each
print("Building KDTree and querying nearest neighbors...")
grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
tree = cKDTree(np.column_stack((x, y)))
_, indices = tree.query(grid_points, k=1)

# Assign nearest z-values
z_filled = z[indices].reshape((grid_size, grid_size))

# Normalize to 16-bit
z_min = np.min(z_filled)
z_max = np.max(z_filled)
z_norm = ((z_filled - z_min) / (z_max - z_min) * 65535).astype(np.uint16)

# Save TIFF
transform = from_origin(xmin, ymax, x_res, y_res)
output_tif = os.path.splitext(csv_path)[0] + f'_nearest_{grid_size}x{grid_size}.tif'

with rasterio.open(
    output_tif, 'w',
    driver='GTiff',
    height=grid_size,
    width=grid_size,
    count=1,
    dtype=z_norm.dtype,
    crs='+proj=latlong',
    transform=transform
) as dst:
    dst.write(z_norm, 1)

print(f"Saved nearest-neighbor TIFF as {output_tif}")

# Save RAW
output_raw = os.path.splitext(csv_path)[0] + f'_nearest_{grid_size}x{grid_size}.raw'
z_norm.tofile(output_raw)
print(f"Saved nearest-neighbor 16-bit RAW as {output_raw}")
