import os
import math
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import distance_transform_edt
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter, median_filter


# Optional: for .ply support
try:
    from plyfile import PlyData
except ImportError:
    PlyData = None

UNITY_SIZES = [33, 65, 129, 257, 513, 1025, 2049, 4097]

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
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 4:
                    data.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return pd.DataFrame(data, columns=['x', 'y', 'z'])

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
    if PlyData is None:
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
    # Z is left unchanged
    return df

def best_unity_grid_size(num_unique_x, num_unique_y):
    max_possible = min(num_unique_x, num_unique_y, 4097)
    for size in reversed(UNITY_SIZES):
        if size <= max_possible:
            return size
    return UNITY_SIZES[0]

def fill_holes_nearest(grid, nodata=65535, max_radius=5):
    mask = (grid == nodata)
    if not np.any(mask):
        return grid
    distances, nearest = distance_transform_edt(mask, return_distances=True, return_indices=True)
    filled = grid.copy()
    fill_mask = (mask) & (distances <= max_radius)
    filled[fill_mask] = grid[tuple(nearest[:, fill_mask])]
    return filled

def rasterize_xyz(df, grid_size, output_folder, tile_x, tile_y):
    x, y, z = df['x'].values, df['y'].values, df['z'].values
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    width = x_max - x_min
    length = y_max - y_min
    min_height = np.min(z)
    max_height = np.max(z)

    # Use binned_statistic_2d to average z values in each cell
    stat, x_edge, y_edge, binnumber = binned_statistic_2d(
        x, y, z, statistic='mean', bins=grid_size, range=[[x_min, x_max], [y_min, y_max]]
    )
    grid = np.full((grid_size, grid_size), 65535, dtype=np.uint16)
    mask = ~np.isnan(stat)
    if max_height != min_height:
        grid[mask] = ((stat[mask] - min_height) / (max_height - min_height) * 65535).astype(np.uint16)
    else:
        grid[mask] = 0

    # Fill holes as before
    grid = fill_holes_nearest(grid, nodata=65535, max_radius=10)

    # Smoothing (optional)
    grid = gaussian_filter(grid.astype(float), sigma=2)
    grid = np.clip(grid, 0, 65535).astype(np.uint16)

    # ROTATE 90 DEGREES LEFT
    grid_to_save = np.rot90(grid, 1)

    # RAW
    raw_name = f"tile_{x_min:.2f}_{y_min:.2f}_{width:.2f}_{length:.2f}_{min_height:.2f}_{max_height:.2f}_rasterized{grid_size}x{grid_size}.raw"
    raw_grid = np.flipud(grid_to_save)
    raw_path = os.path.join(output_folder, raw_name)
    raw_grid.tofile(raw_path)
    print(f"Saved 16-bit RAW as {raw_path}")

    # TIFF
    tiff_name = f"tile_{x_min:.2f}_{y_min:.2f}_{width:.2f}_{length:.2f}_{min_height:.2f}_{max_height:.2f}_rasterized{grid_size}x{grid_size}.tif"
    tiff_path = os.path.join(output_folder, tiff_name)
    x_res = (x_max - x_min) / (grid_size - 1)
    y_res = (y_max - y_min) / (grid_size - 1)
    transform = from_origin(x_min, y_max, x_res, y_res)
    with rasterio.open(
        tiff_path, 'w', driver='GTiff', height=grid_to_save.shape[0], width=grid_to_save.shape[1],
        count=1, dtype=grid_to_save.dtype, crs='+proj=latlong', transform=transform, nodata=65535
    ) as dst:
        dst.write(grid_to_save, 1)
    print(f"Saved rasterized TIFF as {tiff_path}")

    return tiff_path, raw_path

def split_and_rasterize(df, output_prefix):
    x, y = df['x'].values, df['y'].values
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    num_unique_x = len(unique_x)
    num_unique_y = len(unique_y)

    grid_size = best_unity_grid_size(num_unique_x, num_unique_y)
    print(f"Best Unity grid size: {grid_size}")

    n_tiles_x = math.ceil(num_unique_x / grid_size)
    n_tiles_y = math.ceil(num_unique_y / grid_size)

    print(f"Splitting into {n_tiles_x} x {n_tiles_y} tiles...")

    folder = output_prefix
    if not os.path.exists(folder):
        os.makedirs(folder)

    x_sorted = np.sort(unique_x)
    y_sorted = np.sort(unique_y)

    tile_count = 0
    # Loop order: Y outer, X inner (top-to-bottom, then left-to-right)
    for ix in range(n_tiles_x):
        for iy in range(n_tiles_y):
            x_start = ix * grid_size
            x_end = min((ix + 1) * grid_size, num_unique_x)
            y_start = iy * grid_size
            y_end = min((iy + 1) * grid_size, num_unique_y)

            x_min = x_sorted[x_start]
            x_max = x_sorted[x_end - 1]
            y_min = y_sorted[y_start]
            y_max = y_sorted[y_end - 1]

            x_mask = (x >= x_min) & (x <= x_max)
            y_mask = (y >= y_min) & (y <= y_max)
            mask = x_mask & y_mask
            tile_df = df[mask]
            if len(tile_df) == 0:
                continue

            rasterize_xyz(tile_df, grid_size, folder, ix, iy)
            tile_count += 1

     

    print(f"Saved {tile_count} tiles to folder: {folder}")

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select point cloud file",
        filetypes=[("Point cloud files", ".csv .obj .pts .ply .txt"), ("All files", ".*")]
    )
    if not file_path:
        print("No file selected.")
        return

    df = load_xyz(file_path)
    print("Loaded columns:", df.columns)
    df['z'] = df['z'] - 2.0
    df = relative_to_first(df, decimals=2)

    out_csv = os.path.splitext(file_path)[0] + "_relative.csv"
    df.to_csv(out_csv, index=False, float_format='%.2f')
    print(f"Saved relative CSV to {out_csv}")

    output_prefix = os.path.splitext(file_path)[0]
    split_and_rasterize(df, output_prefix)

if __name__ == "__main__":
    main()