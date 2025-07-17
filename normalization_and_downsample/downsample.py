import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

def downsample_csv(input_csv, output_csv, min_distance):
    df = pd.read_csv(input_csv)
    if not set(['x', 'y', 'z']).issubset(df.columns):
        raise ValueError("CSV must have columns: x, y, z")
    points = df[['x', 'y', 'z']].values

    tree = cKDTree(points)
    mask = np.ones(len(points), dtype=bool)

    for i, point in enumerate(points):
        if not mask[i]:
            continue
        idxs = tree.query_ball_point(point, min_distance)
        idxs = [idx for idx in idxs if idx > i]
        mask[idxs] = False

    df_downsampled = df[mask]
    df_downsampled.to_csv(output_csv, index=False)
    return len(df), len(df_downsampled)

def main():
    root = tk.Tk()
    root.withdraw()

    input_csv = filedialog.askopenfilename(
        title="Select input CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not input_csv:
        messagebox.showinfo("Cancelled", "No input file selected.")
        return

    min_distance = simpledialog.askfloat(
        "Minimum Distance",
        "Enter minimum distance between points:",
        initialvalue=0.05,
        minvalue=0.00001
    )
    if min_distance is None:
        messagebox.showinfo("Cancelled", "No distance entered.")
        return

    output_csv = filedialog.asksaveasfilename(
        title="Save downsampled CSV as",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not output_csv:
        messagebox.showinfo("Cancelled", "No output file selected.")
        return

    try:
        before, after = downsample_csv(input_csv, output_csv, min_distance)
        messagebox.showinfo("Success", f"Downsampled from {before} to {after} points.\nSaved to:\n{output_csv}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    main()