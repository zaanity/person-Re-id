import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from reid.dataset import ImageReIDDataset
from reid.model import ReIDNet
from reid.loss import batch_hard_triplet_loss
from reid.utils import plot_pid_dist, collate_with_paths

from re_ranking import re_ranking  # ensure this is on your PYTHONPATH

P, K = 16, 4
MARGIN = 0.3
LR = 1e-4
EPOCHS = 100

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, pids in loader:
        imgs, pids = imgs.to(device), pids.to(device)
        optimizer.zero_grad()
        embeds = model(imgs)
        loss = batch_hard_triplet_loss(embeds, pids, MARGIN)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir',   default='YOUR_TRAIN_DIR')
    parser.add_argument('--query_dir',   default='YOUR_QUERY_DIR')
    parser.add_argument('--gallery_dir', default='YOUR_GALLERY_DIR')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_tf = T.Compose([
        T.Resize((256,128)), 
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2,0.2,0.2,0.1),
        T.ToTensor(),
        T.RandomErasing(p=0.5),
    ])

    train_ds = ImageReIDDataset(args.train_dir, transform=train_tf)
    sampler = torch.utils.data.RandomSampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=P*K, sampler=sampler,
        num_workers=4, pin_memory=True
    )

    plot_pid_dist(train_ds)

    model = ReIDNet(128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {'loss':[]}
    for epoch in range(1, EPOCHS+1):
        loss = train_epoch(model, train_loader, optimizer, device)
        history['loss'].append(loss)
        print(f"Epoch {epoch:02d} — Loss {loss:.4f}")
        torch.save(model.state_dict(), f'logs/model_{epoch:02d}.pth')

if __name__ == '__main__':
    main()
