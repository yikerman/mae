from functools import partial
import os
import glob

import h5py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

from models_mae import MaskedAutoencoderViT
import models_mae

# Updated Statistics for [Temperature, Dfun]
bubbleml_mean = [-48.3421, -2.7237]
bubbleml_std = [105.0279, 2.8301]


class PoolBoilingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.frame_counts = []
        self.cumulative_indices = [0]

        # sorted to ensure consistent ordering
        search_path = os.path.join(root_dir, "PoolBoiling-*", "*.hdf5")
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
