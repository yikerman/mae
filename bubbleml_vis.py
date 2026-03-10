import os
import glob
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

from bubbleml_common import PoolBoilingDataset, medium_bubbleml, ApplyQuantileTransform, DenormalizeQuantile, load_latest_small_model, run_mae_inference

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

    model = load_latest_small_model(checkpoint_dir, device)

    norm_transform = ApplyQuantileTransform(qt_path)
    denormalizer = DenormalizeQuantile(qt_path)
    
    dataset = PoolBoilingDataset(root_dir=data_root, transform=norm_transform)
    
    idx = random.randint(0, len(dataset) - 1)
    print(f"Visualizing sample index: {idx}")
    
    x = dataset[idx][0].unsqueeze(0).to(device)

    loss, y, mask_vis = run_mae_inference(model, x, mask_ratio)
    print(f"loss: {loss}")

    x_unnorm = denormalizer(x)
    y_unnorm = denormalizer(y)
    
    im_masked = x_unnorm * (1 - mask_vis)
    im_paste = x_unnorm * (1 - mask_vis) + y_unnorm * mask_vis

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