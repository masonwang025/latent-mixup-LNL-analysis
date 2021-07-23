import torch
import torch.nn as nn
import numpy as np
import enum


LEAKY_RELU_ALPHA = 0.3


class MixupMode(enum.Enum):
    NO_MIXUP = 'no_mixup'
    INPUT_MIXUP = 'input_mixup'
    MANIFOLD_MIXUP = 'manifold_mixup'


class SpiralModel(nn.Module):
    def __init__(self):
        self.layer1 = nn.Sequential(nn.Linear(1024, 1024), torch.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
        self.layer2 = nn.Sequential(nn.Linear(1024, 1024), torch.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
        self.layer3 = nn.Sequential(nn.Linear(1024, 1024), torch.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
        self.layer4 = nn.Sequential(nn.Linear(1024, 1024), torch.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
        self.layer5 = nn.Sequential(nn.Linear(1024, 1024), torch.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
        self.layer6 = nn.Sequential(nn.Linear(1024, 2), torch.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
    
    def forward(self, x, *, target=None, mixup=False, mixup_hidden=False, mixup_alpha=None, mixup_data=None, device='cuda'):
        if mixup_hidden:
            layer_mix = random.randint(1, 4) # late mixup: random.randint(1, 4)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None

        out = torch.reshape(x, (-1, 1024))

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