import torch.nn as nn
from typing import List
import torch

class PlainMLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(128, 64)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers += [nn.Linear(last, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class NoBNMLPFeature(nn.Module):
    def __init__(self, in_dim: int, hid_dims: List[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        last = in_dim
        for h in hid_dims:
            self.layers.append(nn.Linear(last, h))
            last = h
        self.final_dim = last

    def forward(self, x):
        feats = []
        for lin in self.layers:
            x = torch.relu(lin(x))
            feats.append(x)
        return feats

class DAN_NoBN(nn.Module):
    def __init__(self, in_dim: int, hid_dims: List[int], num_classes: int, mmd_layers: List[int]):
        super().__init__()
        self.backbone = NoBNMLPFeature(in_dim, hid_dims)
        self.classifier = ClassifierHead(self.backbone.final_dim, num_classes)
        self.mmd_layers = mmd_layers

    def forward(self, x_s, x_t):
        feats_s = self.backbone(x_s)
        feats_t = self.backbone(x_t)
        logits_s = self.classifier(feats_s[-1])
        return logits_s, feats_s, feats_t

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2)
        )

    def forward(self, x):
        return self.net(x)