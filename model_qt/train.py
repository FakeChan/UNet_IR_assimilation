import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  
from torch.utils.data import DataLoader
from model import UNet
from data import AMUSADatasetFromJSON
import os

def masked_mse(pred, target, x):

    bt_mask_channels = x[:, -10:, :, :]  # [B, 10, H, W]
   
    spatial_mask = (bt_mask_channels.min(dim=1, keepdim=True)[0] > 0.5).float()  # [B, 1, H, W]
    mask = spatial_mask.expand_as(pred)  # [B, 52, H, W]

    loss = ((pred - target) * mask).pow(2).sum() / (mask.sum() + 1e-8)
    return loss


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            x = batch['input'].to(device, non_blocking=True)
            y = batch['target'].to(device, non_blocking=True)

            pred = model(x)
            loss = masked_mse(pred, y, x)

            if not torch.isfinite(loss):
                print("Warning: non-finite validation loss encountered, skipping this batch.")
                continue

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
    model.train()
    return total_loss / total_samples if total_samples > 0 else float('inf')


if __name__ == '__main__':
    train_json = "/share/home/lililei4/lzx/Unet/code/timelist/time_aligned_triplets_traintime.json"
    val_json   = "/share/home/lililei4/lzx/Unet/code/timelist/time_aligned_triplets_validtime.json"
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = AMUSADatasetFromJSON(train_json)
    val_dataset   = AMUSADatasetFromJSON(val_json)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(in_channels=46, out_channels=26).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0

    model.train()
    for epoch in range(200):
        total_train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            x = batch['input'].to(device, non_blocking=True)
            y = batch['target'].to(device, non_blocking=True)

            pred = model(x)
            loss = masked_mse(pred, y, x)

            if not torch.isfinite(loss):
                print(f"Warning: non-finite training loss at epoch {epoch}, skipping backward.")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else float('inf')
        val_loss = validate(model, val_loader, device)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

        scheduler.step(val_loss)  

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"New best model saved (Val Loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print("Training completed.")