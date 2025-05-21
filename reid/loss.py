import torch
import torch.nn.functional as F

def batch_hard_triplet_loss(embeddings, labels, margin):
    dist = torch.cdist(embeddings, embeddings, p=2)
    N = dist.size(0)
    labels = labels.unsqueeze(1)
    is_pos = (labels == labels.t()) & ~torch.eye(N, device=dist.device).bool()
    is_neg = labels != labels.t()

    hardest_pos = (dist * is_pos.float()).max(dim=1)[0]
    max_dist = dist.max().item()
    dist_neg = dist + max_dist * (~is_neg).float()
    hardest_neg = dist_neg.min(dim=1)[0]

    return F.relu(hardest_pos - hardest_neg + margin).mean()
