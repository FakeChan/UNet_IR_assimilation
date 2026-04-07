import os
import json
from datetime import datetime

amusa_dir = r"/share/home/lililei4/lzx/Unet/data/"
era5_base = r"/share/home/lililei4/lzx/Unet/"
pangu_base = r"/share/home/lililei4/lzx/dataset"   
output_dir = r"/share/home/lililei4/lzx/Unet/code/creat/"

target_time_str = "2024010306"   
dt = datetime.strptime(target_time_str, "%Y%m%d%H")

amusa_file = os.path.join(amusa_dir, f"{dt.strftime('%Y%m%d')}t{dt.strftime('%H')}.npz")
era5_analysis = os.path.join(
    era5_base,
    dt.strftime("%Y"),
    dt.strftime("%m"),
    dt.strftime("%d"),
    dt.strftime("%H") + "00",
    "numpy_output",
    "input_upper.npy"
)
pangu_output_path = os.path.join(
    pangu_base,
    dt.strftime("%Y%m"),
    dt.strftime("%Y%m%d%H") + ".npy"
)

triplet = {
    "valid_time": dt.isoformat(),
    "amusa_file": amusa_file,
    "era5_analysis": era5_analysis,
    "preassim": pangu_output_path
}


output_file = os.path.join(output_dir, f"triplet_{target_time_str}.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump([triplet], f, indent=2, ensure_ascii=False)

print(f"{output_file}")
print(json.dumps([triplet], indent=2))