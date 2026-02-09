import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets as dsets

from cifar_10_adaption import small_cifar10, cifar10_mean, cifar10_std


def load_model(checkpoint_path, device):
    model = small_cifar10()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def get_test_loader(data_root, batch_size):
    normalize = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    test_set = dsets.CIFAR10(root=data_root, train=False, transform=transform_test, download=False)
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )


def extract_latents(model, loader, device):
    latents, masks, ids_restores, labels = [], [], [], []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            latent, mask, ids_restore = model.forward_encoder(images, 0)
            latents.append(latent.cpu())
            masks.append(mask.cpu())
            ids_restores.append(ids_restore.cpu())
            labels.append(target)
    latents = torch.cat(latents)
    masks = torch.cat(masks)
    ids_restores = torch.cat(ids_restores)
    labels = torch.cat(labels)
    return latents, masks, ids_restores, labels


def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    
    torch.manual_seed(3648)

    model = load_model("checkpoints/cifar10_mae_epoch48.pth", device)
    loader = get_test_loader(".", 256)

    latents, masks, ids_restores, labels = extract_latents(model, loader, device)

    payload = {
        "latents": latents,
        "labels": labels,
    }

    print(payload["latents"].shape)

    torch.save(payload, "cifar_latents.pth")


if __name__ == "__main__":
    main()
