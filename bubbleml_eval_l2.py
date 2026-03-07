import random
import torch
import numpy as np
from tqdm import tqdm

from bubbleml_common import PoolBoilingDataset, ApplyQuantileTransform, DenormalizeQuantile, load_latest_small_model, run_mae_inference

def compute_l2(pred, target):
    diff_norm = torch.linalg.vector_norm(pred - target, ord=2)
    return diff_norm.item()
    
def compute_l1(pred, target, eps=1e-8):
    diff_norm = torch.linalg.vector_norm(pred - target, ord=1)
    return diff_norm.item()


def main():
    data_root = "./bubbleml-ds"
    checkpoint_dir = "checkpoints"
    qt_path = "quantile_transform.joblib"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_ratio = 0.75
    num_samples = 512

    model = load_latest_small_model(checkpoint_dir, device)

    norm_transform = ApplyQuantileTransform(qt_path)
    denormalizer = DenormalizeQuantile(qt_path)
    
    dataset = PoolBoilingDataset(root_dir=data_root, transform=norm_transform)
    
    total_available = len(dataset)
    sample_size = min(num_samples, total_available)
    indices = random.sample(range(total_available), sample_size)
    
    l2_errors_temp = []
    l2_errors_dfun = []
    l1_errors_temp = []
    l1_errors_dfun = []

    print(f"Evaluating L2 error over {sample_size} random samples...")

    for idx in tqdm(indices, desc="Computing L2 Errors"):
        x = dataset[idx][0].unsqueeze(0).to(device) 
        
        _, y, mask_vis = run_mae_inference(model, x, mask_ratio)
        
        x_unnorm = denormalizer(x)
        y_unnorm = denormalizer(y)
        im_paste = x_unnorm * (1 - mask_vis) + y_unnorm * mask_vis
        l2_err_temp = compute_l2(im_paste[0, 0], x_unnorm[0, 0])
        l2_err_dfun = compute_l2(im_paste[0, 1], x_unnorm[0, 1])
        l1_err_temp = compute_l1(im_paste[0, 0], x_unnorm[0, 0])
        l1_err_dfun = compute_l1(im_paste[0, 1], x_unnorm[0, 1])

        l2_errors_temp.append(l2_err_temp)
        l2_errors_dfun.append(l2_err_dfun)
        l1_errors_temp.append(l1_err_temp)
        l1_errors_dfun.append(l1_err_dfun)


    print(f"Results over {sample_size} samples:")
    print(f"mean of temp rel L2: {np.mean(l2_errors_temp)}")
    print(f"std of temp rel L2: {np.std(l2_errors_temp)}")
    print(f"mean of temp rel L1: {np.mean(l1_errors_temp)}")
    print(f"std of temp rel L1: {np.std(l1_errors_temp)}")

if __name__ == "__main__":
    main()