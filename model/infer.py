import torch
import torch.nn as nn
from model import UNet
from data import AMUSADatasetFromJSON
import numpy as np
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet(in_channels=72, out_channels=52).to(device)
model.load_state_dict(torch.load(r"/share/home/lililei4/lzx/Unet/code/model/checkpoints/best_model.pth", map_location=device))
model.eval()

# Load dataset
dataset = AMUSADatasetFromJSON(r"/share/home/lililei4/lzx/Unet/code/timelist/time_aligned_triplets_testtime.json")
sample = dataset[1]  

x = sample['input'].unsqueeze(0).to(device)     
y = sample['target'].unsqueeze(0).to(device)   
bt_mask = sample['bt_mask'].unsqueeze(0).to(device)  

# Prediction
with torch.no_grad():
    pred = model(x)

era5_mean = dataset.era5_mean.to(device) 
era5_std  = dataset.era5_std.to(device)
preassim_mean = dataset.preassim_mean.to(device)
preassim_std = dataset.preassim_std.to(device)

pred = pred.view(4, 13, 720, 1440)
y = y.view(4, 13, 720, 1440)

denorm_pred = pred * era5_std + era5_mean
denorm_y = y * era5_std + era5_mean

spatial_mask = bt_mask.all(dim=1, keepdim=True)  # (1,1,720,1440)

# Model RMSE
diff = denorm_pred - denorm_y
squared_error = (diff ** 2) * spatial_mask
sum_se = squared_error.sum(dim=(2, 3))        
count_valid = spatial_mask.sum(dim=(2, 3))      
rmse_map = torch.sqrt(sum_se / (count_valid + 1e-8))

# Baseline RMSE (using input x)
x_52 = x[:, :52, :, :] 
x_52 = x_52.view(4, 13, 720, 1440)
denorm_x = x_52 * preassim_std + preassim_mean

diff_baseline = denorm_x - denorm_y
squared_error_baseline = (diff_baseline ** 2) * spatial_mask  
sum_se_baseline = squared_error_baseline.sum(dim=(2, 3))
rmse_baseline = torch.sqrt(sum_se_baseline / (count_valid + 1e-8))

# Convert to numpy for printing
rmse_baseline_np = rmse_baseline.cpu().numpy()  # shape: (4, 13)
rmse_map_np = rmse_map.cpu().numpy()  

var_names = ['q', 't', 'u', 'v']
layers = list(range(1, 14))

# Print table for each variable
for i, var in enumerate(var_names):
    print(f"\n========== Variable: {var.upper()} ==========")
    print(f"{'Level':<6}{'Baseline RMSE':<16}{'Model RMSE':<16}")
    for level in layers:
        idx = level - 1
        baseline_val = rmse_baseline_np[i, idx]
        model_val = rmse_map_np[i, idx]
        print(f"{level:<6}{baseline_val:<16.6f}{model_val:<16.6f}")

save_dir = r"/share/home/lililei4/lzx/Unet/code/model/results/"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "rmse_baseline.npy"), rmse_baseline_np)
np.save(os.path.join(save_dir, "rmse_model.npy"), rmse_map_np)

print(f"\nRMSE results saved to {save_dir}")