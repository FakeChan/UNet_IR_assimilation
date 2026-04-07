import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
def load_stats(path):
    data = np.load(path)
    return data['mean'], data['std']

class AMUSADatasetFromJSON(Dataset):
    def __init__(self, json_path):

        with open(json_path, 'r') as f:
            self.samples = json.load(f)

        # Load normalization stats once (as tensors)
        bt_mean, bt_std = load_stats(r"/share/home/lililei4/lzx/Unet/code/stauts/bt_15ch.npz")
        preassim_mean, preassim_std = load_stats(r"/share/home/lililei4/lzx/Unet/code/stauts/preassim_stats.npz")
        era5_mean, era5_std = load_stats(r"/share/home/lililei4/lzx/Unet/code/stauts/era5_analysis_stats.npz")
        self.bt_mean = torch.from_numpy(bt_mean[4:14][:, None, None]).float()
        self.bt_std  = torch.from_numpy(bt_std[4:14][:, None, None]).float()
        self.preassim_mean = torch.from_numpy(preassim_mean[1:5,:][:,:,None,None]).float()
        self.preassim_std  = torch.from_numpy(preassim_std[1:5,:][:,:,None,None]).float()
        self.era5_mean = torch.from_numpy(era5_mean[1:5,:][:, :,None, None]).float()
        self.era5_std  = torch.from_numpy(era5_std[1:5,: ][:, :,None, None]).float()


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        bt_data = np.load(item['amusa_file'])

        bt=bt_data["bt_15ch"]              #(15, 721, 1440) -90 90 -180 180
        era5 = np.load(item['era5_analysis'])   # (5, 13,721, 1440) 90 -90 0 360
        preassim = np.load(item['preassim'])  # (5,13, 721, 1440) -90 90 -180 180
        
        era5=np.concatenate([era5[:, :, :, 720:], era5[:, :, :, :720]], axis=-1)#0-360 ---- -180-180
        era5=era5[:,:,::-1,:]
        

        bt = bt.astype(np.float32)
        era5 = era5.astype(np.float32)
        preassim = preassim.astype(np.float32)

        # Select channels
        preassim_sel = preassim[1:5,:,:,:]      # (4,13, H, W)
        bt_sel    = bt[4:14]         # (10, H, W)
        era5_sel  = era5[1:5,:,:,:]         # (4,13, H, W)

        # Handle NaN in BT
        bt_mask = ~np.isnan(bt_sel)  # True = valid (not NaN)
        bt_filled = np.copy(bt_sel)
        bt_mean=self.bt_mean.squeeze().numpy()
        for i in range(bt_sel.shape[0]):
            nan_mask = np.isnan(bt_sel[i])
            bt_filled[i][nan_mask] = bt_mean[i] 

        
        preassim_t = torch.from_numpy(preassim_sel)
        bt_t    = torch.from_numpy(bt_filled)
        era5_t  = torch.from_numpy(era5_sel)
        bt_mask_t = torch.from_numpy(bt_mask).bool()
        mask_channels = bt_mask_t.float()  # (10, 721, 1440)
        preassim_norm = (preassim_t - self.preassim_mean) / (self.preassim_std + 1e-8)
        bt_norm    = (bt_t - self.bt_mean) / (self.bt_std + 1e-8)
        era5_norm  = (era5_t - self.era5_mean) / (self.era5_std + 1e-8)

        
        preassim_flat = preassim_norm.view(4 * 13, 721, 1440)  
        
        input_combined = torch.cat([preassim_flat, bt_norm], dim=0)  
        input_combined = torch.cat([input_combined, mask_channels], dim=0) 
        input_combined = input_combined[:,:720,:]  
        
        era5_flat = era5_norm.view(4 * 13, 721, 1440)
        era5_flat = era5_flat[:,:720,:]
        bt_mask_t = bt_mask_t[:,:720,:]

        return {
            'input': input_combined,      
            'target': era5_flat,       
            'bt_mask': bt_mask_t  
        }
if __name__ == '__main__':
    dataset = AMUSADatasetFromJSON(r"/share/home/lililei4/lzx/Unet/code/timelist/time_aligned_triplets_traintime.json")
    sample = dataset[0]
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    
    print("input shape:", sample['input'].shape)      #  torch.Size([58, 720, 1440])
    print("target shape:", sample['target'].shape)    #  torch.Size([52, 720, 1440])
    print("bt_mask shape:", sample['bt_mask'].shape)  #  torch.Size([10, 720, 1440])
