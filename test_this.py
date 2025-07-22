import os
import math
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from scipy.stats import binned_statistic_2d

# ... (other functions unchanged, e.g., file parsing, relative_to_first, etc.) ...

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

    print(f"Tile {tile_x},{tile_y}: X [{x_min}, {x_max}], Y [{y_min}, {y_max}], Z [{min_height}, {max_height}]")
    print(f"  Unity Terrain settings for this tile:")
    print(f"    Width  (X): {width:.3f}")
    print(f"    Length (Z): {length:.3f}")
    print(f"    Height (Y): {max_height - min_height:.3f}")
    print(f"    Heightmap Resolution: {grid_size}")
    print(f"    Place terrain at X: {x_min:.3f}, Z: {y_min:.3f}, Y: {min_height:.3f}")

    # RAW
    raw_name = f"tile_{x_min:.2f}_{y_min:.2f}_{width:.2f}_{length:.2f}_{min_height:.2f}_{max_height:.2f}_rasterized{grid_size}x{grid_size}.raw"
    raw_path = os.path.join(output_folder, raw_name)
    grid.tofile(raw_path)
    print(f"Saved 16-bit RAW as {raw_path}")

    # TIFF (for checking)
    tiff_name = f"tile_{x_min:.2f}_{y_min:.2f}_{width:.2f}_{length:.2f}_{min_height:.2f}_{max_height:.2f}_rasterized{grid_size}x{grid_size}.tif"
    tiff_path = os.path.join(output_folder, tiff_name)
    x_res = (x_max - x_min) / (grid_size - 1)
    y_res = (y_max - y_min) / (grid_size - 1)
    transform = from_origin(x_min, y_max, x_res, y_res)
    with rasterio.open(
        tiff_path, 'w', driver='GTiff', height=grid.shape[0], width=grid.shape[1],
        count=1, dtype=grid.dtype, crs='+proj=latlong', transform=transform, nodata=65535
    ) as dst:
        dst.write(np.flipud(grid), 1)
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
    df = relative_to_first(df, decimals=2)

    out_csv = os.path.splitext(file_path)[0] + "_relative.csv"
    df.to_csv(out_csv, index=False, float_format='%.2f')
    print(f"Saved relative CSV to {out_csv}")

    output_prefix = os.path.splitext(file_path)[0]
    split_and_rasterize(df, output_prefix)

if __name__ == "__main__":
    main()