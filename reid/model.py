import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ReIDNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(backbone.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)
