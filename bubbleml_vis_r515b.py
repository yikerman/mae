from functools import partial
from types import SimpleNamespace
import os
import glob
import random

import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets as dsets
from torch.utils.data import DataLoader, Dataset

import util.misc as misc
from models_mae import MaskedAutoencoderViT
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from engine_pretrain import train_one_epoch

# Updated Statistics for [Temperature, Dfun]
bubbleml_mean = [-22.1040, -2.6252]
bubbleml_std = [3.8738, 2.7054]

class PoolBoilingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.frame_counts = []
        self.cumulative_indices = [0]
        
        # sorted to ensure consistent ordering
        search_path = os.path.join(root_dir, "PoolBoiling-*-R515B-*", "*.hdf5")
        self.file_paths = sorted(glob.glob(search_path))
        
        print(f"Indexing {len(self.file_paths)} files...")
        total_frames = 0
        for p in self.file_paths:
            with h5py.File(p, 'r') as f:
                # We assume temperature and dfun have the same length
                count = f["temperature"].shape[0]
                self.frame_counts.append(count)
                total_frames += count
                self.cumulative_indices.append(total_frames)
        
        print(f"Total frames indexed: {total_frames}")

    def __len__(self):
        return self.cumulative_indices[-1]

    def __getitem__(self, idx):
        # Find which file the global index belongs to
        file_idx = 0
        for i in range(len(self.cumulative_indices) - 1):
            if idx < self.cumulative_indices[i+1]:
                file_idx = i
                break
        
        # Calculate local frame index within that file
        local_idx = idx - self.cumulative_indices[file_idx]
        file_path = self.file_paths[file_idx]
        
        with h5py.File(file_path, 'r') as f:
            # Load both channels
            temp = f["temperature"][local_idx]
            dfun = f["dfun"][local_idx]
            
            # Stack along channel axis (axis 0)
            # Resulting shape: (2, H, W)
            # Channel 0: Temperature
            # Channel 1: dfun
            data = np.stack([temp, dfun], axis=0).astype('float32')
            
        sample = torch.from_numpy(data)

        if torch.isnan(sample).any():
            print(f"NaN detected in {file_path} at index {local_idx}")
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, 0

def small_bubbleml() -> MaskedAutoencoderViT:
    return models_mae.MaskedAutoencoderViT(
        img_size=64,
        patch_size=8,
        in_chans=2,
        depth=12, embed_dim=512, num_heads=8,
        decoder_depth=8, decoder_num_heads=16, decoder_embed_dim=512,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


def unnormalize(tensor):
    """Reverts the normalization for visualization purposes."""
    mean = torch.tensor(bubbleml_mean).view(1, 2, 1, 1).to(tensor.device)
    std = torch.tensor(bubbleml_std).view(1, 2, 1, 1).to(tensor.device)
    return tensor * std + mean

def save_monochrome_image(tensor_2d, filename):
    """Saves a 2D tensor as a grayscale image, auto-scaling min/max."""
    arr = tensor_2d.detach().cpu().numpy()
    # Using matplotlib to safely handle floating point data (including negatives)
    # and map them smoothly to a 0-255 grayscale color map.
    plt.imsave(filename, arr, cmap='gray')

def main():
    data_root = "./bubbleml-ds"
    checkpoint_dir = "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_ratio = 0.75

    # 1. Initialize Model and Load Weights
    model = small_bubbleml().to(device)
    model.eval() # Set to evaluation mode
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "bubbleml_mae_2ch_R515B_epoch*.pth"))
    if not checkpoint_files:
        print("No checkpoints found in", checkpoint_dir)
        return
        
    # Sort by epoch and load latest
    checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('epoch')[1].split('.pth')[0]))
    latest_checkpoint = checkpoint_files[-1]
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    # 2. Setup Dataset and get a random sample
    normalize = transforms.Normalize(mean=bubbleml_mean, std=bubbleml_std)
    transform_vis = transforms.Compose([normalize])
    dataset = PoolBoilingDataset(root_dir=data_root, transform=transform_vis)
    
    idx = random.randint(0, len(dataset) - 1)
    print(f"Visualizing sample index: {idx}")
    x, _ = dataset[idx]
    
    # Add batch dimension: (1, 2, 64, 64)
    x = x.unsqueeze(0).to(device)

    # 3. Run MAE inference
    with torch.no_grad():
        loss, y, mask = model(x.float(), mask_ratio=mask_ratio)
        y = model.unpatchify(y)
    
    # 4. Process mask for visualization
    p = model.patch_embed.patch_size[0]
    mask_vis = mask.detach()
    # Expand mask to (N, H*W, p*p*in_chans) - Note the *2 for in_chans
    mask_vis = mask_vis.unsqueeze(-1).repeat(1, 1, p**2 * 2) 
    mask_vis = model.unpatchify(mask_vis)  # (1, 2, 64, 64)

    # 5. Unnormalize tensors back to physical values
    x_unnorm = unnormalize(x)
    y_unnorm = unnormalize(y)

    # 6. Reconstruct masked and pasted images
    im_masked = x_unnorm * (1 - mask_vis)
    im_paste = x_unnorm * (1 - mask_vis) + y_unnorm * mask_vis

    # 7. Save outputs for each channel
    channels = ['temp', 'dfun']
    for c, ch_name in enumerate(channels):
        save_monochrome_image(x_unnorm[0, c], f"original_{ch_name}.png")
        save_monochrome_image(im_masked[0, c], f"masked_{ch_name}.png")
        save_monochrome_image(y_unnorm[0, c], f"reconstruction_{ch_name}.png")
        save_monochrome_image(im_paste[0, c], f"reconstruction_paste_{ch_name}.png")
        print(f"Saved {ch_name} visualizations.")

if __name__ == "__main__":
    main()