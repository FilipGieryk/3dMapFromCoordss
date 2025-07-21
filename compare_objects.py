def obj_vertices_set(obj_path):
    """
    Reads an OBJ file and returns a set of rounded vertex tuples.
    """
    vertices = set()
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) == 4:
                    try:
                        # Round to 6 decimal places for floating point tolerance
                        x, y, z = [round(float(coord), 6) for coord in parts[1:]]
                        vertices.add((x, y, z))
                    except ValueError:
                        continue
    return vertices

def compare_obj_files(obj1_path, obj2_path):
    """
    Compares two OBJ files for identical sets of vertices.
    Returns True if they are the same, False otherwise.
    """
    verts1 = obj_vertices_set(obj1_path)
    verts2 = obj_vertices_set(obj2_path)
    return verts1 == verts2

# Example usage:
are_same = compare_obj_files('originalCar.obj', 'output.obj')
print("Same!" if are_same else "Different!")