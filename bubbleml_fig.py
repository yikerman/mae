import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

from bubbleml_common import PoolBoilingDataset, ApplyQuantileTransform, DenormalizeQuantile, load_latest_med_model, run_mae_inference

def plot_n_temp_frames(temp_targets, temp_preds, masks, n, save_path="temp_comparison_n.png"):
    """
    Plots n frames of Temperature Ground Truth, Mask, Prediction (im_paste), and Absolute Error.
    """
    # squeeze=False ensures axarr is always a 2D array, even if n=1
    fig, axarr = plt.subplots(n, 4, figsize=(22, 5 * n), squeeze=False, layout="constrained")

    for i in range(n):
        target = temp_targets[i]
        pred = temp_preds[i]
        mask = masks[i]

        # Calculate shared color scale based on the ground truth
        temp_mean, temp_std = np.mean(target), np.std(target)
        temp_min = temp_mean - 3 * temp_std
        temp_max = temp_mean + 3 * temp_std

        # Calculate absolute error
        temp_err = np.abs(target - pred)
        err_max = np.max(temp_err) if np.max(temp_err) > 0 else 1.0 # Fallback to 1.0 to avoid warnings if error is exactly 0

        # 1. Ground Truth
        im0 = axarr[i, 0].imshow(target, cmap="turbo", vmin=temp_min, vmax=temp_max, origin="lower")
        if i == 0: axarr[i, 0].set_title("Temp Ground Truth")
        axarr[i, 0].axis("off")
        fig.colorbar(im0, ax=axarr[i, 0], fraction=0.04, pad=0.05)

        # 2. Mask
        # Using a gray colormap where 1 (missing/masked) is white, 0 (visible) is black
        im1 = axarr[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1, origin="lower")
        if i == 0: axarr[i, 1].set_title("Mask (White = Masked)")
        axarr[i, 1].axis("off")
        fig.colorbar(im1, ax=axarr[i, 1], fraction=0.04, pad=0.05)

        # 3. Prediction (im_paste)
        im2 = axarr[i, 2].imshow(pred, cmap="turbo", vmin=temp_min, vmax=temp_max, origin="lower")
        if i == 0: axarr[i, 2].set_title("Temp Prediction (im_paste)")
        axarr[i, 2].axis("off")
        fig.colorbar(im2, ax=axarr[i, 2], fraction=0.04, pad=0.05)

        # 4. Absolute Error
        # Updated title and vmax to use the dynamically calculated err_max
        im3 = axarr[i, 3].imshow(temp_err, cmap="turbo", vmin=0, vmax=err_max, origin="lower")
        if i == 0: axarr[i, 3].set_title("Absolute Error")
        axarr[i, 3].axis("off")
        fig.colorbar(im3, ax=axarr[i, 3], fraction=0.04, pad=0.05)

    # Save the plot
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot successfully saved to: {save_path}")
    plt.close()

def main():
    data_root = "./bubbleml-ds"
    checkpoint_dir = "checkpoints"
    qt_path = "quantile_transform.joblib"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_ratio = 0.75
    
    n_samples = 5

    model = load_latest_med_model(checkpoint_dir, device)
    model.eval()

    dataset = PoolBoilingDataset(
        root_dir=data_root, 
        transform=ApplyQuantileTransform(qt_path)
    )
    
    total_available = len(dataset)
    
    indices = random.sample(range(total_available), n_samples)
    print(f"Selected random dataset indices: {indices}")
    
    x_batch = torch.stack([dataset[idx][0] for idx in indices]).to(device)

    with torch.no_grad():
        _, y_batch, mask_vis_batch = run_mae_inference(model, x_batch, mask_ratio)

    unnormalize = DenormalizeQuantile(qt_path)
    
    x_unnorm = unnormalize(x_batch)
    y_unnorm = unnormalize(y_batch)
    
    im_paste = x_unnorm * (1 - mask_vis_batch) + y_unnorm * mask_vis_batch

    temp_targets = x_unnorm[:, 0].detach().cpu().numpy()
    temp_preds = im_paste[:, 0].detach().cpu().numpy()
    masks = mask_vis_batch[:, 0].detach().cpu().numpy()

    save_filename = f"plot_med.png"
    plot_n_temp_frames(temp_targets, temp_preds, masks, n=n_samples, save_path=save_filename)

if __name__ == "__main__":
    main()