with open("input.txt", "r") as infile, open("output.csv", "w") as outfile:
    for line in infile:
        line = line.strip()
        if line:
            parts = line.split()
            outfile.write(",".join(parts) + "\n")