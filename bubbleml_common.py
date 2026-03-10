from functools import partial
import os
import glob

import h5py
import torch
import torch.nn as nn
import numpy as np
import joblib

from models_mae import MaskedAutoencoderViT
import models_mae

# [temp, dfun]
bubbleml_mean = [-48.3421, -2.7237]
bubbleml_std = [105.0279, 2.8301]


class PoolBoilingDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.frame_counts = []
        self.cumulative_indices = [0]

        search_path = os.path.join(root_dir, "PoolBoiling-*", "*.hdf5")
        self.file_paths = sorted(glob.glob(search_path))

        print(f"Indexing {len(self.file_paths)} files...")
        total_frames = 0
        for p in self.file_paths:
            with h5py.File(p, 'r') as f:
                count = f["temperature"].shape[0]
                self.frame_counts.append(count)
                total_frames += count
                self.cumulative_indices.append(total_frames)

        print(f"Total frames indexed: {total_frames}")
        self.file_handles = {}

    def __len__(self):
        return self.cumulative_indices[-1]

    def __getitem__(self, idx):
        file_idx = 0
        for i in range(len(self.cumulative_indices) - 1):
            if idx < self.cumulative_indices[i+1]:
                file_idx = i
                break
        local_idx = idx - self.cumulative_indices[file_idx]
        file_path = self.file_paths[file_idx]

        if file_path not in self.file_handles:
            self.file_handles[file_path] = h5py.File(file_path, 'r')
        f = self.file_handles[file_path]

        temp = f["temperature"][local_idx]
        dfun = f["dfun"][local_idx]

        data = np.stack([temp, dfun], axis=0).astype('float32')
        sample = torch.from_numpy(data)

        if torch.isnan(sample).any():
            print(f"NaN detected in {file_path} at index {local_idx}")

        if self.transform:
            sample = self.transform(sample)

        return sample, 0

    def __del__(self):
        if hasattr(self, 'file_handles'):
            for f in self.file_handles.values():
                f.close()

def small_bubbleml() -> MaskedAutoencoderViT:
    """1/8 compression"""
    return models_mae.MaskedAutoencoderViT(
        img_size=64,
        patch_size=8,
        in_chans=2,
        depth=12, embed_dim=512, num_heads=8,
        decoder_depth=8, decoder_num_heads=16, decoder_embed_dim=512,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True,
    )

def medium_bubbleml() -> MaskedAutoencoderViT:
    """1/4 compression"""
    return models_mae.MaskedAutoencoderViT(
        img_size=64,
        patch_size=8,
        in_chans=2,
        depth=10, embed_dim=1024, num_heads=8,
        decoder_depth=8, decoder_num_heads=16, decoder_embed_dim=512,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True,
    )


class ApplyQuantileTransform:
    """Normalize: Tensor (2, H, W) -> Normalized Tensor (2, H, W)"""
    def __init__(self, qt_path):
        self.qt = joblib.load(qt_path)
        
    def __call__(self, sample):
        c, h, w = sample.shape
        flat = sample.permute(1, 2, 0).reshape(-1, c).numpy()
        normed = self.qt.transform(flat)
        return torch.from_numpy(normed).reshape(h, w, c).permute(2, 0, 1).float()

class DenormalizeQuantile:
    """Unnormalize: Tensor (C, H, W) or (B, C, H, W) -> Original Scale Tensor (C, H, W) or (B, C, H, W)"""
    def __init__(self, qt_path):
        self.qt = joblib.load(qt_path)

    def __call__(self, tensor):
        is_batch = tensor.dim() == 4
        device = tensor.device
        
        if not is_batch:
            tensor = tensor.unsqueeze(0)
        
        b, c, h, w = tensor.shape
        # Permute to (N, C) for sklearn
        flat = tensor.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
        
        # Inverse Transform
        unnormed = self.qt.inverse_transform(flat)
        
        # Reshape to (B, H, W, C) and Permute back to (B, C, H, W)
        unnormed = unnormed.reshape(b, h, w, c)
        unnormed_tensor = torch.from_numpy(unnormed).permute(0, 3, 1, 2).to(device).float()
        
        if not is_batch:
            return unnormed_tensor[0]
        return unnormed_tensor

# todo: remove dupe

def load_latest_med_model(checkpoint_dir, device):
    model = medium_bubbleml().to(device)
    model.eval()
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "bubbleml_mae_2ch_med_epoch*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
    checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('epoch')[1].split('.pth')[0]))
    latest_checkpoint = checkpoint_files[-1]
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])

    return model

def load_latest_small_model(checkpoint_dir, device):
    model = small_bubbleml().to(device)
    model.eval()
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "bubbleml_mae_2ch_epoch*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
    checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('epoch')[1].split('.pth')[0]))
    latest_checkpoint = checkpoint_files[-1]
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])

    return model

def load_latest_meanstd_model(checkpoint_dir, device):
    model = small_bubbleml().to(device)
    model.eval()
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "bubbleml_mae_2ch_meanstd_epoch*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
    checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('epoch')[1].split('.pth')[0]))
    latest_checkpoint = checkpoint_files[-1]
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])

    return model


def run_mae_inference(model, x, mask_ratio):
    """
    Runs inference on the MAE model, unpatchifies the prediction, 
    and formats the mask for visual reconstruction.
    """
    with torch.no_grad():
        loss, y, mask = model(x.float(), mask_ratio=mask_ratio)
        y_unpatchified = model.unpatchify(y)
    
        # Process mask for visualization/pasting
        p = model.patch_embed.patch_size[0]
        mask_vis = mask.detach()
        # Assumes 2 channels (temp and dfun) based on p**2 * 2
        mask_vis = mask_vis.unsqueeze(-1).repeat(1, 1, p**2 * 2) 
        mask_vis = model.unpatchify(mask_vis)
        
    return loss, y_unpatchified, mask_vis

