from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision import datasets as dsets
from torch.utils.data import DataLoader

import util.misc as misc
from models_mae import MaskedAutoencoderViT
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from engine_pretrain import train_one_epoch

cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.247, 0.243, 0.261]

def main():
    # configuration
    data_root = "."
    batch_size = 256
    epochs = 48
    mask_ratio = 0.75
    lr = 1e-3
    weight_decay = 0.05

    # compute setup
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    seed = 3648
    torch.manual_seed(seed)
    np.random.seed(seed)

    # CIFAR-10 augmentation: light crop/flip and standard mean-std
    normalize = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,])

    train_set = dsets.CIFAR10(root=data_root, train=True, transform=transform_train, download=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = small_cifar10().to(device)

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

    for epoch in range(epochs):
        train_stats = train_one_epoch(
            model, train_loader,
            optimizer, device, epoch, scaler,
            args=args
        )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        print(log_stats)
        if epoch % 8 == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), f"checkpoints/cifar10_mae_epoch{epoch+1}.pth")


def small_cifar10() -> MaskedAutoencoderViT:
    return models_mae.MaskedAutoencoderViT(
        # cifar is 32*32*3
        img_size=32,
        patch_size=4,
        in_chans=3,
        # cheap model for testing
        depth=12,
        embed_dim=64,
        decoder_depth=8,
        num_heads=8,
        decoder_embed_dim=64,
        decoder_num_heads=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


if __name__ == "__main__":
    main()
