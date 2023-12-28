"""
Pytorch implementation of Funnel Activation (FReLU): https://arxiv.org/pdf/2007.11824.pdf
"""


import torch
import torch.nn as nn
from typing import Union, Callable, Dict, Optional

class FReLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_aggregation = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1, groups = in_channels)

        self.norm = nn.GroupNorm(32, in_channels)

    def forward(self, x):
        funnel = self.spatial_aggregation(x)
        funnel = self.norm(funnel)

        return torch.max(x, funnel)