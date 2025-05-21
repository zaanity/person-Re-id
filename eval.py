import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from reid.dataset import ImageReIDDataset
from reid.model import ReIDNet
from reid.utils import (
    extract_features, plot_history, plot_cmc, display_retrievals, collate_with_paths
)
from re_ranking import re_ranking  # as before

TOPK = 5

def evaluate(model, q_loader, g_loader, device):
    q_feats, q_pids, q_paths = extract_features(model, q_loader, device)
    g_feats, g_pids, g_paths = extract_features(model, g_loader, device)

    q_q = torch.cdist(q_feats, q_feats).numpy()
    q_g = torch.cdist(q_feats, g_feats).numpy()
    g_g = torch.cdist(g_feats, g_feats).numpy()
    dist_rerank = re_ranking(q_g, q_q, g_g)

    # compute CMC, mAP, retrievals same as before...
    # return cmc_curve, mAP, retrievals, q_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_dir',   default='YOUR_QUERY_DIR')
    parser.add_argument('--gallery_dir', default='YOUR_GALLERY_DIR')
    parser.add_argument('--checkpoint',  required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_tf = T.Compose([T.Resize((256,128)), T.ToTensor()])

    q_ds = ImageReIDDataset(args.query_dir, transform=test_tf, return_paths=True)
    g_ds = ImageReIDDataset(args.gallery_dir, transform=test_tf, return_paths=True)

    q_loader = DataLoader(q_ds, batch_size=64, shuffle=False, collate_fn=collate_with_paths)
    g_loader = DataLoader(g_ds, batch_size=64, shuffle=False, collate_fn=collate_with_paths)

    model = ReIDNet(128).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    cmc, mAP, retrievals, q_paths = evaluate(model, q_loader, g_loader, device)
    print(f"Rank-1: {cmc[0]*100:.2f}%, mAP: {mAP:.2f}%")
    plot_history({'loss':[], 'rank1': [c*100 for c in cmc], 'mAP':[mAP]})
    plot_cmc(cmc, topk=TOPK)
    display_retrievals(retrievals, q_paths, topk=TOPK)

if __name__ == '__main__':
    main()
