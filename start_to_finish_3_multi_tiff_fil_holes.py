import os
import math
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import generic_filter

# Optional: for .ply support
try:
    from plyfile import PlyData
except ImportError:
    PlyData = None

# --- Header Ensurer ---

def ensure_xyz_header(csv_path):
    """
    Ensures the file at csv_path has a header 'x,y,z'.
    If not, prepends it.
    """
    with open(csv_path, "r") as f:
        first_line = f.readline().strip().lower()
        if all(h in first_line for h in ['x', 'y', 'z']):
            return  # Header exists

    with open(csv_path, "r") as infile:
        lines = infile.readlines()
    with open(csv_path, "w") as outfile:
        outfile.write("x,y,z\n")
        outfile.writelines(lines)

# --- File Parsers ---

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

# --- Relative to First Point ---

def relative_to_first(df, decimals=2):
    ref = df.iloc[0][['x', 'y', 'z']].values.astype(float)
    df[['x', 'y', 'z']] = np.round(df[['x', 'y', 'z']].astype(float) - ref, decimals)
    return df

# --- Fill Holes ---

def fill_holes_with_neighbors(grid, nodata=65535, max_iter=10):
    def avg_filter(values):
        center = values[len(values)//2]
        if center != nodata:
            return center
        neighbors = values.copy()
        neighbors = neighbors[neighbors != nodata]
        if len(neighbors) == 0:
            return nodata
        return np.mean(neighbors)
    filled = grid.copy().astype(float)
    for _ in range(max_iter):
        prev_nodata = np.sum(filled == nodata)
        filled = generic_filter(filled, avg_filter, size=3, mode='constant', cval=nodata)
        curr_nodata = np.sum(filled == nodata)
        if curr_nodata == 0 or curr_nodata == prev_nodata:
            break
    return filled.astype(grid.dtype)

# --- Rasterization ---

def rasterize_xyz(df, grid_size=4097, output_prefix=None):
    x, y, z = df['x'].values, df['y'].values, df['z'].values

    grid = np.full((grid_size, grid_size), 65535, dtype=np.uint16)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    x_res = (xmax - xmin) / grid_size
    y_res = (ymax - ymin) / grid_size

    valid_min, valid_max = np.min(z), np.max(z)
    if valid_max == valid_min:
        z_scaled = np.full_like(z, 0, dtype=np.uint16)
    else:
        z_scaled = ((z - valid_min) / (valid_max - valid_min) * 65535).astype(np.uint16)

    cols = ((x - xmin) / x_res).astype(int)
    rows = ((ymax - y) / y_res).astype(int)
    cols = np.clip(cols, 0, grid_size - 1)
    rows = np.clip(rows, 0, grid_size - 1)

    grid[rows, cols] = z_scaled

    # Fill holes with neighbor average
    grid = fill_holes_with_neighbors(grid, nodata=65535)

    # Save TIFF
    transform = from_origin(xmin, ymax, x_res, y_res)
    tiff_path = f"{output_prefix}_rasterized{grid_size}x{grid_size}.tif"
    with rasterio.open(
        tiff_path, 'w', driver='GTiff', height=grid.shape[0], width=grid.shape[1],
        count=1, dtype=grid.dtype, crs='+proj=latlong', transform=transform, nodata=65535
    ) as dst:
        dst.write(grid, 1)
    print(f"Saved rasterized TIFF as {tiff_path}")

    # Save RAW
    raw_path = f"{output_prefix}_rasterized{grid_size}x{grid_size}.raw"
    grid.tofile(raw_path)
    print(f"Saved 16-bit RAW as {raw_path}")

    return tiff_path, raw_path

# --- Split and Rasterize ---

def split_and_rasterize(df, output_prefix, max_grid_size=4097):
    x, y = df['x'].values, df['y'].values

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # Use unique values to estimate density
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    n_tiles_x = math.ceil(len(unique_x) / max_grid_size)
    n_tiles_y = math.ceil(len(unique_y) / max_grid_size)

    if n_tiles_x == 1 and n_tiles_y == 1:
        rasterize_xyz(df, grid_size=max_grid_size, output_prefix=output_prefix)
        return

    print(f"Splitting into {n_tiles_x} x {n_tiles_y} tiles...")

    folder = output_prefix
    if not os.path.exists(folder):
        os.makedirs(folder)

    x_edges = np.linspace(xmin, xmax, n_tiles_x + 1)
    y_edges = np.linspace(ymin, ymax, n_tiles_y + 1)

    tile_count = 0
    for ix in range(n_tiles_x):
        for iy in range(n_tiles_y):
            x0, x1 = x_edges[ix], x_edges[ix+1]
            y0, y1 = y_edges[iy], y_edges[iy+1]

            if ix == n_tiles_x - 1:
                x_mask = (x >= x0) & (x <= x1)
            else:
                x_mask = (x >= x0) & (x < x1)
            if iy == n_tiles_y - 1:
                y_mask = (y >= y0) & (y <= y1)
            else:
                y_mask = (y >= y0) & (y < y1)
            mask = x_mask & y_mask
            tile_df = df[mask]
            if len(tile_df) == 0:
                continue

            tile_prefix = os.path.join(folder, f"{os.path.basename(output_prefix)}_tile_{ix}_{iy}")
            rasterize_xyz(tile_df, grid_size=max_grid_size, output_prefix=tile_prefix)
            tile_count += 1

    print(f"Saved {tile_count} tiles to folder: {folder}")

# --- Main GUI and Workflow ---

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
    df = relative_to_first(df, decimals=2)

    out_csv = os.path.splitext(file_path)[0] + "_relative.csv"
    df.to_csv(out_csv, index=False, float_format='%.2f')
    print(f"Saved relative CSV to {out_csv}")

    output_prefix = os.path.splitext(file_path)[0]
    split_and_rasterize(df, output_prefix, max_grid_size=4097)

if __name__ == "__main__":
    main()