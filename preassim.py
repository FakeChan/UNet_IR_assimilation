import os
import json
import numpy as np

# Configuration: update these paths for your Linux environment
json_path = r"/share/home/lililei4/lzx/Unet/code/timelist/time_aligned_triplets_traintime.json"

# Load triplet mapping
with open(json_path, "r", encoding="utf-8") as f:
    triplets = json.load(f)

# Expected data dimensions
n_vars = 5      # Z, Q, T, U, V
n_levels = 13   # pressure levels

# Initialize accumulators with high precision
total_sum = np.zeros((n_vars, n_levels), dtype=np.float64)
total_sum_sq = np.zeros((n_vars, n_levels), dtype=np.float64)
total_count = np.zeros((n_vars, n_levels), dtype=np.int64)

processed = 0

for item in triplets:
    era5_path = item["preassim"]
    if not os.path.exists(era5_path):
        print(f"Warning: File not found, skipping: {era5_path}")
        continue

    try:
        data = np.load(era5_path)  # Expected shape: (5, 13, 721, 1440)
        if data.shape != (5, 13, 721, 1440):
            print(f"Warning: Shape mismatch, skipping: {era5_path}to{data.shape}")
            continue

        data = data.astype(np.float64)

        for v in range(n_vars):
            for lev in range(n_levels):
                layer_data = data[v, lev].ravel()  # flatten spatial dimensions
                valid_mask = ~np.isnan(layer_data)
                valid_values = layer_data[valid_mask]

                total_sum[v, lev] += np.sum(valid_values)
                total_sum_sq[v, lev] += np.sum(valid_values ** 2)
                total_count[v, lev] += valid_values.size

        processed += 1
        print(f"Processed: {os.path.basename(era5_path)} | Total: {processed}")

    except Exception as e:
        print(f"Error loading {era5_path}: {e}")

if processed == 0:
    raise RuntimeError("No valid forecast files were processed!")

# Compute global mean and standard deviation
global_mean = np.divide(
    total_sum, total_count,
    out=np.zeros_like(total_sum),
    where=(total_count > 0)
)

variance = np.divide(
    total_sum_sq, total_count,
    out=np.zeros_like(total_sum_sq),
    where=(total_count > 0)
) - global_mean ** 2

global_std = np.sqrt(np.clip(variance, 0, None))

# Handle degenerate cases (zero std, NaN, inf)
global_std = np.where((global_std == 0) | ~np.isfinite(global_std), 1.0, global_std)
global_mean = np.where(~np.isfinite(global_mean), 0.0, global_mean)

# Print results
variables = ["Z", "Q", "T", "U", "V"]
print("\n" + "="*70)
print("Global statistics for Pangu 24h forecast (by variable x pressure level)")
print("="*70)
for v in range(n_vars):
    print(f"\nVariable: {variables[v]}")
    print("Level\tMean\t\tStd")
    for lev in range(n_levels):
        print(f"{lev}\t{global_mean[v, lev]:.4f}\t\t{global_std[v, lev]:.4f}")

# Save statistics
save_dir = r"/share/home/lililei4/lzx/Unet/code/stauts/"
save_path = os.path.join(save_dir, "preassim_stats.npz")
np.savez(save_path, mean=global_mean, std=global_std, count=total_count)
print(f"\nStatistics saved to: {save_path}")