import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------------------------------------------------------------
# Configuration
root_dir = "/share/home/lililei4/lzx/Unet/code/creat/result"
pressure_levels = ['1000', '925', '850', '700', '600', '500', '400',
                   '300', '250', '200', '150', '100', '50']
# Variable names (adjust if needed)
var_names = ['q', 't', 'u', 'v']

# ----------------------------------------------------------------------
# Collect and sort all time?stamped folders that contain rmse_model.npy
folders = []
for f in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, f)
    if os.path.isdir(folder_path) and os.path.isfile(os.path.join(folder_path, "rmse_model.npy")):
        folders.append(f)

# Sort folders by their timestamp (assuming format YYYY-MM-DD_HH-MM-SS)
folders.sort(key=lambda x: datetime.strptime(x, "%Y-%m-%d_%H-%M-%S"))

# ----------------------------------------------------------------------
# Read all RMSE data
data_list = []          # each element: (n_vars, n_levels) = (4,13)
time_labels = []        # store original folder names as labels
for folder in folders:
    file_path = os.path.join(root_dir, folder, "rmse_model.npy")
    data = np.load(file_path)       # shape (4,13)
    data_list.append(data)
    time_labels.append(folder)
    print(f"Loaded: {file_path}")

n_files = len(data_list)
if n_files == 0:
    print("No files found. Exiting.")
    exit()

# Convert list to a 3D array (n_files, 4, 13)
all_data = np.array(data_list)   # shape (n_files, 4, 13)

# ----------------------------------------------------------------------
# Plotting
for var_idx in range(4):
    plt.figure(figsize=(10, 6))
    for level_idx in range(13):
        rmse_series = all_data[:, var_idx, level_idx]
        plt.plot(range(n_files), rmse_series, label=pressure_levels[level_idx])

    # Customize x?axis: show only a few labels to keep it readable
    if n_files <= 10:
        plt.xticks(range(n_files), time_labels, rotation=45, ha='right')
    else:
        # Show first, middle, and last labels
        step = max(1, n_files // 5)   # roughly 5 ticks
        indices = list(range(0, n_files, step))
        # Ensure the last index is included
        if indices[-1] != n_files-1:
            indices.append(n_files-1)
        plt.xticks(indices, [time_labels[i] for i in indices], rotation=45, ha='right')

    plt.xlabel("Time (folder name)")
    plt.ylabel("RMSE")
    plt.title(f"RMSE evolution for variable: {var_names[var_idx]}")
    plt.legend(title="Pressure (hPa)", loc='upper right', ncol=2, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save the plot
    out_file = os.path.join(r"/share/home/lililei4/lzx/Unet/code/creat/plot/", f"rmse_evolution_{var_names[var_idx]}.png")
    plt.savefig(out_file, dpi=150)
    print(f"Saved plot: {out_file}")
    plt.close()

print("All plots generated successfully.")