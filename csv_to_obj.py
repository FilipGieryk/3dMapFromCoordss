def csv_to_obj(csv_path, obj_path):
    """
    Reads a CSV file with x, y, z coordinates and writes an OBJ file with those vertices.
    No interpolation or mesh generation is performed.
    """
    with open(csv_path, 'r') as csv_file, open(obj_path, 'w') as obj_file:
        for line in csv_file:
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue  # skip malformed lines
            try:
                x, y, z = map(float, parts)
                obj_file.write(f"v {x} {y} {z}\n")
            except ValueError:
                continue  # skip lines with non-numeric data

# Example usage:
csv_to_obj('relative_output.csv', 'output.obj')