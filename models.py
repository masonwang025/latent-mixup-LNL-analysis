import torch
import torch.nn as nn
import numpy as np


class SpiralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(2, 1024), torch.nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(1024, 1024), torch.nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 1024), torch.nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(1024, 1024), torch.nn.LeakyReLU())
        self.layer5 = nn.Sequential(
            nn.Linear(1024, 1024), torch.nn.LeakyReLU())
        self.layer6 = nn.Sequential(nn.Linear(1024, 2), torch.nn.LeakyReLU())

    def forward(self, x, *, target=None, layer_mix=None, mixup_alpha=None, mixup_data=None, device='cuda'):
        out = x.float()

        if layer_mix == 0:
            out, targets_a, targets_b, lam, index = mixup_data(
                out, target, alpha=mixup_alpha, device=device)

        out = self.layer1(out)

        if layer_mix == 1:
            out, targets_a, targets_b, lam, index = mixup_data(
                out, target, alpha=mixup_alpha, device=device)

        out = self.layer2(out)

        if layer_mix == 2:
            out, targets_a, targets_b, lam, index = mixup_data(
                out, target, alpha=mixup_alpha, device=device)

        out = self.layer3(out)

        if layer_mix == 3:
            out, targets_a, targets_b, lam, index = mixup_data(
                out, target, alpha=mixup_alpha, device=device)

        out = self.layer4(out)

        if layer_mix == 4:
            out, targets_a, targets_b, lam, index = mixup_data(
                out, target, alpha=mixup_alpha, device=device)

        out = self.layer5(out)

        if layer_mix == 5:
            out, targets_a, targets_b, lam, index = mixup_data(
                out, target, alpha=mixup_alpha, device=device)

        out = self.layer6(out)

        if target is not None:
            return out, targets_a, targets_b, lam, index
        else:
            return out
