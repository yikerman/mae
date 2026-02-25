from cifar_10_adaption import small_cifar10, cifar10_std, cifar10_mean

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

cifar10_std = np.array(cifar10_std)
cifar10_mean = np.array(cifar10_mean)

def save_array_as_image(arr, filename):
    arr = arr.numpy()
    arr = (arr * cifar10_std) + cifar10_mean  # denormalize
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(filename)

# load model
model = small_cifar10()
checkpoint = torch.load("checkpoints/cifar10_mae_epoch48.pth", map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# load and prepare image
img = Image.open("airplane.png")
img = img.resize((32, 32))
img = img.convert('RGB')
img = np.array(img) / 255.
assert img.shape == (32, 32, 3)
img = (img - cifar10_mean) / cifar10_std

# copy pasted
x = torch.tensor(img)

# make it a batch-like
x = x.unsqueeze(dim=0)
x = torch.einsum('nhwc->nchw', x)

# run MAE
loss, y, mask = model(x.float(), mask_ratio=0.75)
y = model.unpatchify(y)
y = torch.einsum('nchw->nhwc', y).detach().cpu()

# visualize the mask
mask = mask.detach()
mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

x = torch.einsum('nchw->nhwc', x)
# masked image
im_masked = x * (1 - mask)

# MAE reconstruction pasted with visible patches
im_paste = x * (1 - mask) + y * mask

# save images
save_array_as_image(x[0], "original.png")
save_array_as_image(im_masked[0], "masked.png")
save_array_as_image(y[0], "reconstruction.png")
save_array_as_image(im_paste[0], "reconstruction_paste.png")

print(loss)
