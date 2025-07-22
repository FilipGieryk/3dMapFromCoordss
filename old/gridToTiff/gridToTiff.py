import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
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

# Adjust if needed
x = df['x'].values
y = df['y'].values
z = df['z'].values

# Grid resolution
grid_size = 513
print(f"Grid size set to {grid_size} x {grid_size} (no interpolation)")

# Create blank grid filled with 65535 (white = no data)
grid = np.full((grid_size, grid_size), 65535, dtype=np.uint16)

# Get grid spacing
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
x_res = (xmax - xmin) / grid_size
y_res = (ymax - ymin) / grid_size

# Normalize z to 16-bit
valid_min = np.min(z)
valid_max = np.max(z)
z_scaled = ((z - valid_min) / (valid_max - valid_min) * 65535).astype(np.uint16)

# Compute row/col indices for each point
cols = ((x - xmin) / x_res).astype(int)
rows = ((ymax - y) / y_res).astype(int)

# Clip indices to avoid out-of-bounds
cols = np.clip(cols, 0, grid_size - 1)
rows = np.clip(rows, 0, grid_size - 1)

# Fill grid (last value wins if duplicate)
grid[rows, cols] = z_scaled

# Save TIFF
transform = from_origin(xmin, ymax, x_res, y_res)
output_tif = os.path.splitext(csv_path)[0] + f'_rasterized_{grid_size}x{grid_size}.tif'

with rasterio.open(
    output_tif, 'w',
    driver='GTiff',
    height=grid.shape[0],
    width=grid.shape[1],
    count=1,
    dtype=grid.dtype,
    crs='+proj=latlong',
    transform=transform,
    nodata=65535
) as dst:
    dst.write(grid, 1)

print(f"Saved rasterized (non-interpolated) TIFF as {output_tif}")

# Save as RAW
output_raw = os.path.splitext(csv_path)[0] + f'_rasterized_{grid_size}x{grid_size}.raw'
grid.tofile(output_raw)
print(f"Saved 16-bit RAW as {output_raw}")
