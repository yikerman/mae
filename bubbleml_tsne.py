import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

# Import from your custom modules
from bubbleml_common import PoolBoilingDataset, ApplyQuantileTransform, load_latest_small_model

def main():
    # --- Configuration ---
    data_root = "./bubbleml-ds"
    checkpoint_dir = "checkpoints"
    qt_path = "quantile_transform.joblib"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_samples = 1024
    batch_size = 64  # Adjust if you run into GPU memory issues
    save_path = "tsne_cls_tokens.png"

    # --- Load Model & Dataset ---
    print("Loading model...")
    model = load_latest_small_model(checkpoint_dir, device)
    model.eval()

    print("Loading dataset...")
    dataset = PoolBoilingDataset(
        root_dir=data_root, 
        transform=ApplyQuantileTransform(qt_path)
    )

    total_available = len(dataset)
    print(f"Total available samples: {total_available}")

    # 1. Sample 1000 datapoints
    # We use Subset and DataLoader to handle batched inference safely
    indices = random.sample(range(total_available), num_samples)
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    cls_tokens = []

    print("Extracting CLS tokens without masking...")
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)
            
            # 2. Run inference on the encoder with mask_ratio=0
            # forward_encoder typically returns (latent, mask, ids_restore)
            latent, mask, ids_restore = model.forward_encoder(x.float(), mask_ratio=0.0)
            
            # 3. Extract the output of cls_token
            # The CLS token is prepended to the patch sequence, making it index 0
            cls_token_batch = latent[:, 0, :]
            cls_tokens.append(cls_token_batch.cpu())

    # Concatenate all batches into a single array
    cls_tokens_tensor = torch.cat(cls_tokens, dim=0)
    cls_tokens_np = cls_tokens_tensor.numpy()

    # 4. Run t-SNE and save plot
    print(f"Running t-SNE on feature shape {cls_tokens_np.shape} (this might take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(cls_tokens_np)

    print("Generating plot...")
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7, edgecolors='w', s=45, cmap='turbo')
    plt.title(f"t-SNE of MAE CLS Tokens\n({num_samples} samples, No Masking)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Plot successfully saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()