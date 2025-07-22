import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import tkinter as tk
from tkinter import filedialog

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
        # Check if header is present
        if all(h in first_line for h in ['x', 'y', 'z']):
            return  # Header exists

    # If not, add header
    with open(csv_path, "r") as infile:
        lines = infile.readlines()
    with open(csv_path, "w") as outfile:
        outfile.write("x,y,z\n")
        outfile.writelines(lines)

# --- File Parsers ---

def parse_csv(path):
    ensure_xyz_header(path)
    df = pd.read_csv(path)
    # Try to find x, y, z columns (case-insensitive)
    cols = [c.lower() for c in df.columns]
    xyz = [cols.index('x'), cols.index('y'), cols.index('z')]
    df = df.iloc[:, xyz]
    df.columns = ['x', 'y', 'z']
    return df

def parse_obj(path):
    # Only reads vertices (lines starting with 'v ')
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 4:
                    data.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return pd.DataFrame(data, columns=['x', 'y', 'z'])

def parse_pts(path):
    # Assumes format: x y z [other columns...]
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
    # Check if file contains commas
    with open(txt_path, "r") as infile:
        first_line = infile.readline()
        has_comma = ',' in first_line
    csv_path = os.path.splitext(txt_path)[0] + ".csv"
    if has_comma:
        # Already CSV-like, just copy and check header
        with open(txt_path, "r") as infile, open(csv_path, "w") as outfile:
            for line in infile:
                outfile.write(line)
        ensure_xyz_header(csv_path)
        return csv_path
    # Otherwise, convert space/tab to comma
    with open(txt_path, "r") as infile, open(csv_path, "w") as outfile:
        lines = infile.readlines()
        # Check if first line is header
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
                parts = line.split()  # splits on any whitespace
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
        # Convert txt to csv, then parse
        csv_path = convert_txt_to_csv(path)
        return parse_csv(csv_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# --- Relative to First Point ---

def relative_to_first(df, decimals=2):
    ref = df.iloc[0][['x', 'y', 'z']].values.astype(float)
    df[['x', 'y', 'z']] = np.round(df[['x', 'y', 'z']].astype(float) - ref, decimals)
    return df

# --- Rasterization ---

def rasterize_xyz(df, grid_size=513, output_prefix=None):
    x, y, z = df['x'].values, df['y'].values, df['z'].values

    # Grid
    grid = np.full((grid_size, grid_size), 65535, dtype=np.uint16)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    x_res = (xmax - xmin) / grid_size
    y_res = (ymax - ymin) / grid_size

    # Normalize z to 16-bit
    valid_min, valid_max = np.min(z), np.max(z)
    z_scaled = ((z - valid_min) / (valid_max - valid_min) * 65535).astype(np.uint16)

    # Compute indices
    cols = ((x - xmin) / x_res).astype(int)
    rows = ((ymax - y) / y_res).astype(int)
    cols = np.clip(cols, 0, grid_size - 1)
    rows = np.clip(rows, 0, grid_size - 1)

    grid[rows, cols] = z_scaled

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

# --- Main GUI and Workflow ---

def pick_grid_size(num_points, unity_sizes=[33, 65, 129, 257, 513, 1025, 2049, 4097]):
    """
    Picks the smallest Unity grid size that is >= sqrt(num_points).
    """
    import math
    min_size = math.ceil(np.sqrt(num_points))
    for size in unity_sizes:
        if size >= min_size:
            return size
    return unity_sizes[-1]  # Use largest if too many points

def main():
    # File dialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select point cloud file",
        filetypes=[("Point cloud files", ".csv .obj .pts .ply .txt"), ("All files", ".*")]
    )
    if not file_path:
        print("No file selected.")
        return

    # Load and process
    df = load_xyz(file_path)
    print("Loaded columns:", df.columns)
    df = relative_to_first(df, decimals=2)

    # Save relative CSV
    out_csv = os.path.splitext(file_path)[0] + "_relative.csv"
    df.to_csv(out_csv, index=False, float_format='%.2f')
    print(f"Saved relative CSV to {out_csv}")

    # Choose grid size
    unity_sizes = [33, 65, 129, 257, 513, 1025, 2049, 4097]
    num_points = len(df)
    grid_size = pick_grid_size(num_points, unity_sizes)
    print(f"Auto-selected grid size: {grid_size} (for {num_points} points)")

    # Optionally, ask user for grid size
    # grid_size = int(input(f"Enter grid size from {unity_sizes}: ") or 513)
    if grid_size not in unity_sizes:
        raise ValueError("Grid size must be one of Unity RAW standard sizes.")

    # Rasterize and save
    output_prefix = os.path.splitext(file_path)[0]
    rasterize_xyz(df, grid_size=grid_size, output_prefix=output_prefix)

if __name__ == "__main__":
    main()