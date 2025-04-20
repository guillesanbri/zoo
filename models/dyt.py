import torch
import torch.nn as nn


class DyT(nn.Module):
    def __init__(self, C, init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.beta = nn.Parameter(torch.ones(C))
        self.gamma = nn.Parameter(torch.ones(C))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta