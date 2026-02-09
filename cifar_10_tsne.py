import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

payload = torch.load('cifar_latents.pth')
features = payload['latents'].view(10000, -1).numpy()
labels = payload['labels'].numpy()

# print("pca")
# pca = PCA(n_components=256)
# features_reduced = pca.fit_transform(features)

print("tsne")
tsne = TSNE(n_components=2, perplexity=50, learning_rate='auto', max_iter=2000, random_state=42)
tsne_results = tsne.fit_transform(features)

plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    x=tsne_results[:, 0],
    y=tsne_results[:, 1],
    hue=labels,
    palette=sns.color_palette("hls", 10),
    legend="full",
    alpha=0.6
)

plt.show()
