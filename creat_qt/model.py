import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_ch=96, bilinear=True):
        super().__init__()
        assert in_channels >= out_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_ch = base_ch
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_ch * 8, base_ch * 16 // factor)

        # Decoder
        self.up1 = Up(base_ch * 16, base_ch * 8 // factor, bilinear)
        self.up2 = Up(base_ch * 8, base_ch * 4 // factor, bilinear)
        self.up3 = Up(base_ch * 4, base_ch * 2 // factor, bilinear)
        self.up4 = Up(base_ch * 2, base_ch, bilinear)

        self.delta_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        base = x[:, :self.out_channels, :, :]  # (B, 52, H, W)

        
        bt_mask_channels = x[:, -10:, :, :]  # (B, 10, H, W)

        
        spatial_mask = (bt_mask_channels.min(dim=1, keepdim=True)[0] > 0.5).float()  

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        delta = self.delta_conv(x)  
       
        delta = delta * spatial_mask  

        return base + delta