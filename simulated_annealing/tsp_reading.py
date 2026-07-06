import numpy as np

def tsp_reading(file_name):
    coords = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    # Check if the file has TSPLIB headers
    has_coords_section = any("NODE_COORD_SECTION" in line for line in lines)
    
    if has_coords_section:
        in_coord_section = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("NODE_COORD_SECTION"):
                in_coord_section = True
                continue
            if line.startswith("EOF"):
                break
            if in_coord_section:
                parts = line.split()
                if len(parts) >= 3:
                    coords.append([float(parts[1]), float(parts[2])])
        return np.array(coords)
    else:
        return np.loadtxt(file_name)