import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def collate_with_paths(batch):
    imgs, pids, paths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    pids = torch.tensor(pids, dtype=torch.long)
    return imgs, pids, list(paths)

def extract_features(model, loader, device):
    model.eval()
    feats, pids, paths = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs, lbls, img_paths = batch
            embeds = model(imgs.to(device)).cpu()
            feats.append(embeds)
            pids.extend(lbls.numpy())
            paths.extend(img_paths)
    return torch.cat(feats), np.array(pids), paths

def plot_pid_dist(dataset):
    counts = {}
    for _, pid in dataset.samples:
        counts[pid] = counts.get(pid, 0) + 1
    plt.figure(figsize=(6,4))
    plt.hist(list(counts.values()), bins=30)
    plt.title('Images per ID')
    plt.xlabel('Number of images'); plt.ylabel('Count')
    plt.show()

def plot_history(history):
    epochs = np.arange(1, len(history['loss'])+1)
    for key in ['loss','rank1','mAP']:
        plt.figure(figsize=(6,4))
        plt.plot(epochs, history[key], 'o-')
        plt.title(f'{key.capitalize()} over Epochs')
        plt.xlabel('Epoch'); plt.ylabel(key.capitalize())
        plt.show()

def plot_cmc(cmc_curve, topk=5):
    ranks = np.arange(1, topk+1)
    plt.figure(figsize=(6,4))
    plt.plot(ranks, cmc_curve[:topk]*100, 'o-')
    plt.title('CMC Curve'); plt.xlabel('Rank'); plt.ylabel('%')
    plt.show()

def display_retrievals(retrievals, query_paths, topk=5):
    for i in range(min(5, len(retrievals))):
        query = Image.open(query_paths[i]).convert('RGB')
        plt.figure(figsize=(12,3))
        plt.subplot(1, topk+1, 1); plt.imshow(query); plt.title('Query'); plt.axis('off')
        for j, img in enumerate(retrievals[i]):
            plt.subplot(1, topk+1, j+2)
            plt.imshow(img); plt.title(f'Retrieval {j+1}'); plt.axis('off')
        plt.show()
