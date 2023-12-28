"""
Pytorch implementation of BioNet: https://openaccess.thecvf.com//content/CVPR2023/papers/Li_BioNet_A_Biologically-Inspired_Network_for_Face_Recognition_CVPR_2023_paper.pdf
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .frelu import FReLU
from .cbam import CBAM


class CFCComponent(nn.Module):
    def __init__(self, in_channels:int,  out_channels:int,) -> None:
        super().__init__()
        self.layers = nn.Sequential(
                                CBAM(in_channels),
                                nn.AdaptiveAvgPool2d((1,1)),
                                nn.Flatten(),
                                nn.Linear(in_channels, out_channels)
        )


class CFC(nn.Module):
    '''
    Cortex Functional Compartmentalization
    '''
    def __init__(self, in_channels:int, out_channels:int, n_attributes:int = 4) -> None:
        super().__init__()
        self.identity_comp = CFCComponent(in_channels, out_channels)
        self.attribute_comps = nn.ModuleDict({
            f'attr_{i}': CFCComponent(in_channels, out_channels) for i in range(n_attributes)
        })

    def forward(self, x):
        id_feat = self.identity_comp(x)

        attr_feat = {
            k: module(x) for k, module in self.attribute_comps.items()
        }

        return id_feat, attr_feat
