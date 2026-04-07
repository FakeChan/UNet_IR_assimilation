import os
import json
import numpy as np
from datetime import datetime

# ----------------------------------------------------------------------
# Configuration
root_dir = "/share/home/lililei4/lzx/Unet/code/creat_qt/result"
json_path = "/share/home/lililei4/lzx/Unet/code/timelist/time_aligned_triplets_testtime.json"

# Pressure levels (for reference only, not used in computation)
pressure_levels = ['1000', '925', '850', '700', '600', '500', '400',
                   '300', '250', '200', '150', '100', '50']

# ----------------------------------------------------------------------
# Load JSON file
with open(json_path, 'r') as f:
    data = json.load(f)   # Expecting a list of dicts

# Extract valid_time strings and convert to folder name format
folder_names = []
for entry in data:
    valid_time = entry.get("valid_time")
    if valid_time:
        
        folder_name = valid_time.replace('T', '_').replace(':', '-')
        folder_names.append(folder_name)
    else:
        print(f"Warning: entry missing 'valid_time'  skipped")

# ----------------------------------------------------------------------
# Initialize accumulator and counter
sum_arr = None
file_count = 0

# Loop over folder names and load corresponding rmse_model.npy
for folder in folder_names:
    folder_path = os.path.join(root_dir, folder)
    file_path = os.path.join(folder_path, "rmse_model.npy")
    if os.path.isfile(file_path):
        data_arr = np.load(file_path)   # Expected shape: (2,13)
        if sum_arr is None:
            sum_arr = data_arr.copy()
        else:
            sum_arr += data_arr
        file_count += 1
        print(f"Loaded: {file_path}")
    else:
        print(f"File not found: {file_path}")

# ----------------------------------------------------------------------
# Compute and save average
if file_count > 0:
    avg_arr = sum_arr / file_count
    print(f"\nProcessed {file_count} file(s). Average array shape: {avg_arr.shape}")

    output_path = os.path.join("/share/home/lililei4/lzx/Unet/code/creat_qt/", "avg_test.npy")
    np.save(output_path, avg_arr)
    print(f"Averaged result saved to: {output_path}")

    print("\nAverage RMSE (2 variables  13 pressure levels):")
    print(avg_arr)
else:
    print("No valid rmse_model.npy files found for the given timestamps.")