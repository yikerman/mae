import torch
import h5py
import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer
import joblib

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
                # assume temperature and dfun have the same shape
                count = f["temperature"].shape[0]
                self.frame_counts.append(count)
                total_frames += count
                self.cumulative_indices.append(total_frames)
        print(f"Total frames indexed: {total_frames}")

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
        with h5py.File(file_path, 'r') as f:
            temp = f["temperature"][local_idx]
            dfun = f["dfun"][local_idx]
            data = np.stack([temp, dfun], axis=0).astype('float32')
        sample = torch.from_numpy(data)
        if torch.isnan(sample).any():
            print(f"NaN detected in {file_path} at index {local_idx}")
        if self.transform:
            sample = self.transform(sample)
        return sample

def generate_and_save_quantile_transformer(loader, save_path="quantile_transform.joblib", max_samples=10_000_000):
    """
    Samples pixels from the dataset and fits a QuantileTransformer.
    max_samples: The maximum number of pixels to collect to avoid memory overload.
                 5 million pixels is more than enough to capture the distribution accurately.
    """
    sampled_pixels = []
    total_collected = 0
    
    print(f"Collecting up to {max_samples:,} pixel samples to fit QuantileTransformer...")
    
    for batch in tqdm(loader, desc="Sampling data"):
        # batch shape: [B, 2, H, W]
        b, c, h, w = batch.shape
        
        # Permute to [B, H, W, 2] and then flatten to [N, 2] where N = B*H*W
        # This treats each pixel as an independent sample with 2 features (temp, dfun)
        flattened = batch.permute(0, 2, 3, 1).reshape(-1, c).numpy()
        
        # Randomly subsample 10% of pixels from this batch to ensure we draw from 
        # a wide variety of frames before hitting max_samples
        n_pixels = flattened.shape[0]
        subsample_size = int(n_pixels * 0.1)
        
        idx = np.random.choice(n_pixels, size=subsample_size, replace=False)
        subsampled_flat = flattened[idx]
        
        sampled_pixels.append(subsampled_flat)
        total_collected += subsampled_flat.shape[0]
        
        if total_collected >= max_samples:
            break

    # Concatenate all collected samples into one large array of shape (N, 2)
    print("Concatenating sampled data...")
    all_data = np.concatenate(sampled_pixels, axis=0)
    
    # Trim exactly to max_samples if we overshot
    if all_data.shape[0] > max_samples:
        all_data = all_data[:max_samples]
        
    print(f"Final fitting shape: {all_data.shape}")
    
    # Fit the QuantileTransformer
    # output_distribution='normal' maps the data to a standard Gaussian (mean 0, std 1)
    # which is ideal for deep learning models.
    print("Fitting QuantileTransformer (this might take a minute)...")
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=2000, random_state=42)
    qt.fit(all_data)
    
    print(f"Saving transformer to {save_path}...")
    joblib.dump(qt, save_path)
    print("Done!")

if __name__ == "__main__":
    # 1. Initialize Dataset without any transforms for data collection
    dataset = PoolBoilingDataset(root_dir='./bubbleml-ds')
    
    # 2. Use a shuffled Dataloader so our samples are uniformly distributed
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    # 3. Generate and save the fit transformer
    generate_and_save_quantile_transformer(loader, save_path="quantile_transform.joblib")
    
    # --- Example of how to use it afterwards ---
    # print("\nTesting the applied transform...")
    # transform = ApplyQuantileTransform("quantile_transform.joblib")
    # dataset_normalized = PoolBoilingDataset(root_dir='./bubbleml-ds', transform=transform)
    # sample = dataset_normalized[0]
    # print(f"Normalized sample shape: {sample.shape}")
    # print(f"Channel 0 mean: {sample[0].mean():.3f}, std: {sample[0].std():.3f}")
    # print(f"Channel 1 mean: {sample[1].mean():.3f}, std: {sample[1].std():.3f}")