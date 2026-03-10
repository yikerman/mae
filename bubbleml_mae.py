from types import SimpleNamespace
import os
import glob

import numpy as np
import torch
import joblib
from torchvision import transforms
from torch.utils.data import DataLoader

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_pretrain import train_one_epoch
from bubbleml_common import PoolBoilingDataset, small_bubbleml, ApplyQuantileTransform

def main():
    # configuration
    data_root = "./bubbleml-ds"
    batch_size = 512
    epochs = 128
    mask_ratio = 0.75
    lr = 1e-5
    weight_decay = 0.05
    checkpoint_dir = "checkpoints"
    qt_path = "quantile_transform.joblib"
    filename_base = "bubbleml_mae_2ch_epoch"

    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    seed = 3648
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    transform_train = transforms.Compose([
        ApplyQuantileTransform(qt_path),
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

    # resume from checkpoint
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f"{filename_base}*.pth"))
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('epoch')[1].split('.pth')[0]))
        latest_checkpoint = checkpoint_files[-1]
        print(f"Found checkpoint: {latest_checkpoint}. Resuming...")
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Successfully loaded full state. Resuming from epoch {start_epoch}.")            
    else:
        print("No checkpoints found. Starting training from scratch.")

    # training
    for epoch in range(start_epoch, epochs):
        train_stats = train_one_epoch(
            model, train_loader,
            optimizer, device, epoch, scaler,
            args=args
        )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        print(log_stats)
        
        # save checkpoint
        save_path = os.path.join(checkpoint_dir, f"{filename_base}{epoch+1}.pth")
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'args': args
        }
        torch.save(save_dict, save_path)
        print(f"Checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()