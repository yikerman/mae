import os
import glob
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

from bubbleml_common import PoolBoilingDataset, small_bubbleml, ApplyQuantileTransform, DenormalizeQuantile

def save_monochrome_image(arr_2d, filename):
    """Saves a 2D numpy array as a grayscale image, auto-scaling min/max."""
    # Using matplotlib to safely handle floating point data and map to 0-255
    plt.imsave(filename, arr_2d, cmap='gray')


def save_monochrome_image(arr_2d, filename):
    """Saves a 2D numpy array as a grayscale image, auto-scaling min/max."""
    plt.imsave(filename, arr_2d, cmap='gray')

def main():
    data_root = "./bubbleml-ds"
    checkpoint_dir = "checkpoints"
    qt_path = "quantile_transform.joblib"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_ratio = 0.75

    # 0. Check for Quantile Transformer
    if not os.path.exists(qt_path):
        print(f"Error: {qt_path} not found. Please run generate_quantile_transform.py first.")
        return

    # 1. Initialize Model and Load Weights
    model = small_bubbleml().to(device)
    model.eval()
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "bubbleml_mae_2ch_epoch*.pth"))
    if not checkpoint_files:
        print("No checkpoints found in", checkpoint_dir)
        return
        
    checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('epoch')[1].split('.pth')[0]))
    latest_checkpoint = checkpoint_files[-1]
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    # 2. Setup Dataset and Denormalizer
    norm_transform = ApplyQuantileTransform(qt_path)
    denormalizer = DenormalizeQuantile(qt_path)
    
    dataset = PoolBoilingDataset(root_dir=data_root, transform=norm_transform)
    
    idx = random.randint(0, len(dataset) - 1)
    print(f"Visualizing sample index: {idx}")
    x = dataset[idx][0] # (2, 64, 64) - normalized tensor
    
    # Add batch dimension: (1, 2, 64, 64)
    x = x.unsqueeze(0).to(device)

    # 3. Run MAE inference
    with torch.no_grad():
        loss, y, mask = model(x.float(), mask_ratio=mask_ratio)
        print(f"loss: {loss}")
        y = model.unpatchify(y)
    
    # 4. Process mask for visualization
    p = model.patch_embed.patch_size[0]
    mask_vis = mask.detach()
    mask_vis = mask_vis.unsqueeze(-1).repeat(1, 1, p**2 * 2) 
    mask_vis = model.unpatchify(mask_vis) # (1, 2, 64, 64)

    # 5. Denormalize back to physical values
    # These now return Tensors in (B, C, H, W)
    x_unnorm = denormalizer(x) 
    y_unnorm = denormalizer(y)
    
    # 6. Reconstruct masked and pasted images in physical units
    im_masked = x_unnorm * (1 - mask_vis)
    im_paste = x_unnorm * (1 - mask_vis) + y_unnorm * mask_vis

    # 7. Save outputs for each channel
    channels = ['temp', 'dfun']
    for c, ch_name in enumerate(channels):
        # x_unnorm is (1, 2, 64, 64), [0, c] gives (64, 64)
        save_monochrome_image(x_unnorm[0, c].cpu().numpy(), f"original_{ch_name}.png")
        save_monochrome_image(im_masked[0, c].cpu().numpy(), f"masked_{ch_name}.png")
        save_monochrome_image(y_unnorm[0, c].cpu().numpy(), f"reconstruction_{ch_name}.png")
        save_monochrome_image(im_paste[0, c].cpu().numpy(), f"reconstruction_paste_{ch_name}.png")
        print(f"Saved {ch_name} visualizations.")

if __name__ == "__main__":
    main()