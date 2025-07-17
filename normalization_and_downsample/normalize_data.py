import csv

def normalize_csv(input_path, output_path):
    with open(input_path, newline='') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty.")

    # Get the first row's coordinates
    x0 = float(rows[0]['x'])
    y0 = float(rows[0]['y'])
    z0 = float(rows[0]['z'])

    # Normalize all rows
    normalized_rows = []
    for row in rows:
        x = float(row['x']) - x0
        y = float(row['y']) - y0
        z = float(row['z']) - z0
        normalized_rows.append({
        'x': round(x, 6),
        'y': round(y, 6),
        'z': round(z, 6)
    })

    # Write to output file
    with open(output_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['x', 'y', 'z'])
        writer.writeheader()
        writer.writerows(normalized_rows)

    print(f"✅ Normalized data written to: {output_path}")

# Example usage
normalize_csv('coords.csv', 'normalized_output.csv')