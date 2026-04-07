import torch
import torch.nn as nn
from model import UNet
from data import AMUSADatasetFromJSON
import numpy as np
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = UNet(in_channels=72, out_channels=52).to(device)
model.load_state_dict(torch.load(r"/share/home/lililei4/lzx/Unet/code/model/checkpoints/best_model.pth", map_location=device))
model.eval()


json_path = r"/share/home/lililei4/lzx/Unet/code/timelist/time_aligned_triplets_alltime.json"
dataset = AMUSADatasetFromJSON(json_path)


with open(json_path, 'r') as f:
    json_list = json.load(f) 


era5_mean = dataset.era5_mean.to(device)
era5_std  = dataset.era5_std.to(device)
preassim_mean = dataset.preassim_mean.to(device)
preassim_std = dataset.preassim_std.to(device)


for idx, meta in enumerate(json_list):
    valid_time = meta['valid_time']  
    
    folder_name = valid_time.replace(':', '-').replace('T', '_')
    sample_dir = os.path.join(r"/share/home/lililei4/lzx/Unet/code/creat/result/", folder_name)
    os.makedirs(sample_dir, exist_ok=True)

    
    sample = dataset[idx] 
    x = sample['input'].unsqueeze(0).to(device)      # (1,72,720,1440)
    y = sample['target'].unsqueeze(0).to(device)     # (1,52,720,1440)
    bt_mask = sample['bt_mask'].unsqueeze(0).to(device)  # (1,13,720,1440)

    with torch.no_grad():
        pred = model(x)  # (1,52,720,1440)

    
    pred = pred.view(4, 13, 720, 1440)   # (4,13,720,1440)
    y = y.view(4, 13, 720, 1440)

   
    denorm_pred = pred * era5_std + era5_mean
    denorm_y = y * era5_std + era5_mean

   
    spatial_mask = bt_mask.all(dim=1, keepdim=True)  # (1,1,720,1440)

   
    diff = denorm_pred - denorm_y
    squared_error = (diff ** 2) * spatial_mask
    sum_se = squared_error.sum(dim=(2, 3))          # (4,13)
    count_valid = spatial_mask.sum(dim=(2, 3))      # (1,1)
    rmse_model = torch.sqrt(sum_se / (count_valid + 1e-8))

    
    x_52 = x[:, :52, :, :]  # (1,52,720,1440)
    x_52 = x_52.view(4, 13, 720, 1440)
    denorm_x = x_52 * preassim_std + preassim_mean
    diff_baseline = denorm_x - denorm_y
    squared_error_baseline = (diff_baseline ** 2) * spatial_mask
    sum_se_baseline = squared_error_baseline.sum(dim=(2, 3))
    rmse_baseline = torch.sqrt(sum_se_baseline / (count_valid + 1e-8))

    
    rmse_model_np = rmse_model.cpu().numpy()        # (4,13)
    rmse_baseline_np = rmse_baseline.cpu().numpy()  # (4,13)
    denorm_pred_np = denorm_pred.cpu().numpy()      # (4,13,720,1440)
    denorm_y_np = denorm_y.cpu().numpy()            # (4,13,720,1440)
    spatial_mask_np = spatial_mask.squeeze().cpu().numpy()  # (720,1440)

   
    np.save(os.path.join(sample_dir, "rmse_baseline.npy"), rmse_baseline_np)
    np.save(os.path.join(sample_dir, "rmse_model.npy"), rmse_model_np)
    np.save(os.path.join(sample_dir, "denorm_pred.npy"), denorm_pred_np)
    np.save(os.path.join(sample_dir, "denorm_y.npy"), denorm_y_np)
    np.save(os.path.join(sample_dir, "spatial_mask.npy"), spatial_mask_np)

    
    var_names = ['q', 't', 'u', 'v']
    layers = list(range(1, 14))
    print(f"\n========== Sample: {valid_time} ==========")
    for i, var in enumerate(var_names):
        print(f"Variable: {var.upper()}")
        print(f"{'Level':<6}{'Baseline RMSE':<16}{'Model RMSE':<16}")
        for level in layers:
            idx_l = level - 1
            baseline_val = rmse_baseline_np[i, idx_l]
            model_val = rmse_model_np[i, idx_l]
            print(f"{level:<6}{baseline_val:<16.6f}{model_val:<16.6f}")

    print(f"Results saved to {sample_dir}")

print("\nAll samples processed.")