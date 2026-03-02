from functools import partial
from types import SimpleNamespace
import os
import glob

import h5py
import torch
import torch.nn as nn
import numpy as np
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

def main():
    # configuration
    data_root = "./bubbleml-ds"
    batch_size = 256
    epochs = 48
    mask_ratio = 0.75
    lr = 1e-4
    weight_decay = 0.1
    checkpoint_dir = "checkpoints"

    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    seed = 3648
    torch.manual_seed(seed)
    np.random.seed(seed)

    normalize = transforms.Normalize(mean=bubbleml_mean, std=bubbleml_std)
    
    transform_train = transforms.Compose([
        normalize,
    ])

    train_set = PoolBoilingDataset(root_dir=data_root, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = small_bubbleml().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = NativeScaler()
    
    args = SimpleNamespace(**{
        "mask_ratio": mask_ratio, 
        "accum_iter": 1, 
        "warmup_epochs": 3, 
        "lr": lr,
        "min_lr": 1e-6,
        "epochs": epochs,
        "blr": lr,})

    start_epoch = 0

    # --- Resume from Checkpoint Logic ---
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "bubbleml_mae_2ch_R515B_epoch*.pth"))
    if checkpoint_files:
        # Sort files by epoch number
        checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('epoch')[1].split('.pth')[0]))
        latest_checkpoint = checkpoint_files[-1]
        
        print(f"Found checkpoint: {latest_checkpoint}. Resuming...")
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            print(f"Successfully loaded full state. Resuming from epoch {start_epoch}.")
        else:
            model.load_state_dict(checkpoint)
            start_epoch = int(os.path.basename(latest_checkpoint).split('epoch')[1].split('.pth')[0])
            print(f"Warning: Loaded old format checkpoint. Resuming from epoch {start_epoch}.")
            
    else:
        print("No checkpoints found. Starting training from scratch.")

    # --- Training Loop ---
    for epoch in range(start_epoch, epochs):
        train_stats = train_one_epoch(
            model, train_loader,
            optimizer, device, epoch, scaler,
            args=args
        )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        print(log_stats)
        
        # Save comprehensive checkpoint
        save_path = os.path.join(checkpoint_dir, f"bubbleml_mae_2ch_R515B_epoch{epoch+1}.pth")
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'args': args
        }
        torch.save(save_dict, save_path)
        print(f"Checkpoint saved to {save_path}")


def small_bubbleml() -> MaskedAutoencoderViT:
    return models_mae.MaskedAutoencoderViT(
        img_size=64,
        patch_size=8,
        in_chans=2,
        depth=12, embed_dim=512, num_heads=8,
        decoder_depth=8, decoder_num_heads=16, decoder_embed_dim=512,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


if __name__ == "__main__":
    main()