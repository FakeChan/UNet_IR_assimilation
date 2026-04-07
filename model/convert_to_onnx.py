import torch
import onnx
from model import UNet   # Make sure model.py is in your Python path

def convert_pth_to_onnx(pth_path, onnx_path, in_channels=72, out_channels=52,
                        height=720, width=1440):
    device = torch.device('cpu')
    model = UNet(in_channels=in_channels, out_channels=out_channels).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    print("Model loaded successfully, switched to eval mode")

    # Create dummy input with fixed spatial dimensions
    dummy_input = torch.randn(1, in_channels, height, width, device=device)
    print(f"Dummy input shape: {dummy_input.shape}")

    # Only batch dimension is dynamic; spatial dimensions are fixed
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,          # Use recent opset for better compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    print(f"ONNX model exported to: {onnx_path}")

    # Validate the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")

if __name__ == '__main__':
    pth_file = "/share/home/lililei4/lzx/Unet/code/model/checkpoints/best_model.pth"
    onnx_file = "/share/home/lililei4/lzx/Unet/code/model/checkpoints/best_model.onnx"
    convert_pth_to_onnx(pth_file, onnx_file,
                        in_channels=72,
                        out_channels=52,
                        height=720,
                        width=1440)