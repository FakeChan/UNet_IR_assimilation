import os
import json
import numpy as np

# Configuration (update these paths for your Linux system)
data_dir = r"/share/home/lililei4/lzx/Unet/data/"
json_path = r"/share/home/lililei4/lzx/Unet/code/timelist/time_aligned_triplets_traintime.json"
n_channels = 15

# Initialize accumulators (use float64 to avoid precision loss)
total_sum = np.zeros(n_channels, dtype=np.float64)      # sum of values
total_sum_sq = np.zeros(n_channels, dtype=np.float64)   # sum of squares
total_count = np.zeros(n_channels, dtype=np.int64)      # count of valid (non-NaN) samples

# Load JSON triplet file
with open(json_path, "r", encoding="utf-8") as f:
    triplets = json.load(f)

# Extract unique AMUSA file paths
amusa_files = list({item["amusa_file"] for item in triplets})
print(f"Processing statistics from {len(amusa_files)} AMUSA files (from {json_path})")

file_count = 0

for fpath in amusa_files:
    if not os.path.exists(fpath):
        print(f"Warning: File not found, skipping: {fpath}")
        continue

    try:
        with np.load(fpath) as data:
            arr = data['bt_15ch']  # expected key in .npz

        if arr.shape[0] != n_channels:
            print(f"Warning: Skipping {os.path.basename(fpath)}: channel count mismatch ({arr.shape[0]} != {n_channels})")
            continue

        arr = arr.astype(np.float64)

        for ch in range(n_channels):
            channel_flat = arr[ch].ravel()
            valid_mask = ~np.isnan(channel_flat)
            valid_data = channel_flat[valid_mask]

            total_sum[ch] += np.sum(valid_data)
            total_sum_sq[ch] += np.sum(valid_data * valid_data)
            total_count[ch] += valid_data.size

        file_count += 1
        print(f"Processed {os.path.basename(fpath)} | Total: {file_count}/{len(amusa_files)}")

    except Exception as e:
        print(f"Error processing {os.path.basename(fpath)}: {e}")

if file_count == 0:
    raise RuntimeError("No valid AMUSA files were loaded!")

# Compute global mean and standard deviation
global_mean = np.divide(total_sum, total_count,
                        out=np.zeros_like(total_sum),
                        where=(total_count > 0))

variance = np.divide(total_sum_sq, total_count,
                     out=np.zeros_like(total_sum_sq),
                     where=(total_count > 0)) - global_mean**2
global_std = np.sqrt(np.clip(variance, 0, None))

# Handle degenerate cases (zero std, NaN, inf)
global_std = np.where((global_std == 0) | ~np.isfinite(global_std), 1.0, global_std)
global_mean = np.where(~np.isfinite(global_mean), 0.0, global_mean)

# Print results
print("\n" + "="*70)
print(f"Global statistics computed from {file_count} AMUSA files (NaN values ignored)")
print("="*70)
print(f"{'Chan':<6} {'Mean':<12} {'Std':<12} {'Valid Samples'}")
print("-"*70)
for ch in range(n_channels):
    print(f"{ch:<6} {global_mean[ch]:<12.4f} {global_std[ch]:<12.4f} {total_count[ch]:,}")

# Save statistics
save_dir = r"/share/home/lililei4/lzx/Unet/code/stauts/"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "bt_15ch.npz")
np.savez(save_path, mean=global_mean, std=global_std, count=total_count)
print(f"\nStatistics saved to: {save_path}")