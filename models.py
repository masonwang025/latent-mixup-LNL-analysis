import torch
import torch.nn as nn
import numpy as np

LEAKY_RELU_ALPHA = 0.3

class SpiralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(2, 256), torch.nn.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
        self.layer2 = nn.Sequential(nn.Linear(256, 1024), torch.nn.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
        self.layer3 = nn.Sequential(nn.Linear(1024, 1024), torch.nn.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
        self.layer4 = nn.Sequential(nn.Linear(1024, 1024), torch.nn.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
        self.layer5 = nn.Sequential(nn.Linear(1024, 1024), torch.nn.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
        self.layer6 = nn.Sequential(nn.Linear(1024, 2), torch.nn.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
    
    def forward(self, x, *, target=None, mixup=False, mixup_hidden=False, mixup_alpha=None, mixup_data=None, device='cuda'):
        if mixup_hidden:
            layer_mix = np.random.randint(0, 4)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None
        
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