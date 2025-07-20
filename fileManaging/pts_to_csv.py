import csv

def load_pts_file(pts_path):
    points = []
    with open(pts_path, 'r') as f:
        lines = f.readlines()

    # Skip first line if it looks like a count, not coordinates
    try:
        float(lines[0].strip().split()[0])
        start = 0
    except ValueError:
        start = 1

    for line in lines[start:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            coords = list(map(float, parts[:3]))  # Support 2D or 3D
            points.append(coords)
    return points

def write_csv(points, csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['X', 'Y', 'Z'])  # Header
        for p in points:
            # Fill with 0.0 if 2D point
            while len(p) < 3:
                p.append(0.0)
            writer.writerow(p)

def pts_to_csv(pts_path, csv_path):
    points = load_pts_file(pts_path)
    write_csv(points, csv_path)
    print(f"Converted {len(points)} points to {csv_path}")

# Example usage
if __name__ == "__main__":
    pts_file = "your_file.pts"  # Replace with your .pts file
    csv_file = "converted.csv"
    pts_to_csv(pts_file, csv_file)
