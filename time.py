import os
from datetime import datetime, timedelta
import json
import random


amusa_dir = r"/share/home/lililei4/lzx/Unet/data/"
era5_base = r"/share/home/lililei4/lzx/Unet/"
pangu_base = r"/share/home/lililei4/lzx/dataset"   

output_dir = r"/share/home/lililei4/lzx/Unet/code/timelist/"


def parse_amusa_time(filename):
    if not filename.endswith(".npz"):
        return None
    base = filename[:-4]
    if 't' not in base:
        return None
    date_str, hour_str = base.split('t')
    if len(date_str) != 8 or len(hour_str) != 2:
        return None
    try:
        return datetime.strptime(date_str + hour_str, "%Y%m%d%H")
    except ValueError:
        return None

# Collect valid AMUSA timestamps
valid_times = set()
amusa_path_by_time = {}

for fname in os.listdir(amusa_dir):
    dt = parse_amusa_time(fname)
    if dt:
        fpath = os.path.join(amusa_dir, fname)
        valid_times.add(dt)
        amusa_path_by_time[dt] = fpath

valid_times = sorted(valid_times)
print(f"Found {len(valid_times)} AMUSA time points")


triplets = []

for vt in valid_times:
    amusa_path = amusa_path_by_time.get(vt)
    if not amusa_path:
        continue

    era5_input_path = os.path.join(
        era5_base,
        vt.strftime("%Y"),
        vt.strftime("%m"),
        vt.strftime("%d"),
        vt.strftime("%H") + "00",
        "numpy_output", "input_upper.npy"
    )

    init_time = vt
    pangu_output_path = os.path.join(
        pangu_base,
        init_time.strftime("%Y%m"),
        init_time.strftime("%Y%m%d%H") + ".npy"
    )

    triplets.append({
        "valid_time": vt.isoformat(),
        "amusa_file": amusa_path,
        "era5_analysis": era5_input_path,
        "preassim": pangu_output_path
    })

# Check existence for first 5 samples
print("\nChecking file existence (first 5 samples):")
for item in triplets[:5]:
    vt = item["valid_time"]
    amusa_ok = os.path.exists(item["amusa_file"])
    era5_ok = os.path.exists(item["era5_analysis"])
    pangu_ok = os.path.exists(item["preassim"])
    status = f"AMUSA:{'OK' if amusa_ok else 'MISS'} | ERA5:{'OK' if era5_ok else 'MISS'} | Pangu24h:{'OK' if pangu_ok else 'MISS'}"
    print(f"{vt} -> {status}")

# Shuffle and split (80% train, 10% val, 10% test)
random.seed(42)
random.shuffle(triplets)

n = len(triplets)
n_train = int(0.8 * n)
n_valid = int(0.1 * n)
n_test = n - n_train - n_valid

train_triplets = triplets[:n_train]
valid_triplets = triplets[n_train:n_train + n_valid]
test_triplets = triplets[n_train + n_valid:]

# Save to JSON
alltime_path = os.path.join(output_dir, "time_aligned_triplets_alltime.json")
train_path = os.path.join(output_dir, "time_aligned_triplets_traintime.json")
valid_path = os.path.join(output_dir, "time_aligned_triplets_validtime.json")
test_path = os.path.join(output_dir, "time_aligned_triplets_testtime.json")

with open(alltime_path, "w", encoding="utf-8") as f:
    json.dump(triplets, f, indent=2, ensure_ascii=False)

with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_triplets, f, indent=2, ensure_ascii=False)

with open(valid_path, "w", encoding="utf-8") as f:
    json.dump(valid_triplets, f, indent=2, ensure_ascii=False)

with open(test_path, "w", encoding="utf-8") as f:
    json.dump(test_triplets, f, indent=2, ensure_ascii=False)

print(f"\nGenerated {len(triplets)} aligned triplets (with 24h Pangu forecasts)")
print("Saved to:")
print(f"  All:     {alltime_path}")
print(f"  Train:   {train_path} ({len(train_triplets)} samples)")
print(f"  Valid:   {valid_path} ({len(valid_triplets)} samples)")
print(f"  Test:    {test_path} ({len(test_triplets)} samples)")