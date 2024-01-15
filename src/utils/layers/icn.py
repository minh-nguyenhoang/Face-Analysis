"""
Pytorch implementation of BioNet: https://openaccess.thecvf.com//content/CVPR2023/papers/Li_BioNet_A_Biologically-Inspired_Network_for_Face_Recognition_CVPR_2023_paper.pdf
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .frelu import FReLU
from .cbam import CBAM
from .dran_attn import DRAN


class CFCComponent(nn.Module):
    '''
    Using CBAM for attribute feature embedding for efficiency.
    '''
    def __init__(self, in_channels:int,  out_channels:int, is_identity = True) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if not is_identity:
            self.layers = nn.Sequential(
                                    CBAM(in_channels),
                                    nn.Conv2d(in_channels, in_channels, kernel_size= 7, groups = in_channels, bias= False),
                                    nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(),
                                    nn.Linear(in_channels, out_channels)
            )
        else:
            self.layers = nn.Sequential(
                                    CBAM(in_channels),
                                    nn.Conv2d(in_channels, in_channels, kernel_size= 7, groups = in_channels, bias= False),
                                    nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(),
                                    nn.Linear(in_channels, out_channels)
            )            
    
    def forward(self, x):
        return self.layers(x)


class CFC(nn.Module):
    '''
    Cortex Functional Compartmentalization
    '''
    def __init__(self, in_channels:int, out_channels:int, n_attributes:int = 4) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_attributes = n_attributes

        self.identity_comp = CFCComponent(in_channels, out_channels, is_identity= True)
        self.attribute_comps = nn.ModuleDict({
            f'attr_{i}': CFCComponent(in_channels, out_channels, is_identity= False) for i in range(n_attributes)
        })

    def forward(self, x):
        id_feat = self.identity_comp(x)

        attr_feat = {
            k: module(x) for k, module in self.attribute_comps.items()
        }

        return id_feat, attr_feat
    


class CRTComponent(nn.Module):
    def __init__(self, in_channels:int, inter_channels:int=None ,) -> None:
        super().__init__()
        if inter_channels is None:
            inter_channels = inter_channels //4

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        self.layers = nn.Sequential(
                                nn.Linear(in_channels, inter_channels),
                                nn.BatchNorm1d(inter_channels),
                                nn.ReLU(),
                                nn.Linear(inter_channels, in_channels),
        )
    
    def forward(self, x):
        return self.layers(x)

class CRT(nn.Module):
    '''
    Compartment Response Transform
    '''
    def __init__(self, in_channels:int, inter_channels:int=None, n_attributes:int = 4) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.inter_channelss = inter_channels
        self.n_attributes = n_attributes

        self.attribute_comps = nn.ModuleDict({
            f'attr_{i}': CRTComponent(in_channels, inter_channels) for i in range(n_attributes)
        })    

    def forward(self, x: dict):
        attr_feat = {
            k: module(inp_feat) for (k, module), inp_feat in zip(self.attribute_comps.items(),x.values())
        }

        return attr_feat
    

class RIM(nn.Module):
    '''
    Response Intensity Modulation
    '''
    def __init__(self, in_channels:int, n_attributes:int = 4) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.n_attributes = n_attributes

        self.attn = nn.Sequential(
                                nn.Linear(in_channels*n_attributes, in_channels*n_attributes//2),
                                nn.BatchNorm1d(in_channels*n_attributes//2),
                                nn.ReLU(),
                                nn.Linear(in_channels*n_attributes//2, n_attributes),
                                nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, attr_dict: dict):

        B, D = x.shape

        attr_feat: torch.Tensor = torch.stack(attr_dict.values(), dim= -2)

        attn_feat: torch.Tensor = self.attn(attr_feat.view(B,-1))

        attr_sum = (attr_feat * attn_feat).sum(-2)

        out = x * self.n_attributes + attr_sum

        return out


class ICN(nn.Module):
    '''
    Inferotemporal Cortex Network
    '''
    def __init__(self, in_channels:int, out_channels: int=None, n_attributes:int = 4) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_attributes = n_attributes   

        self.cfc = CFC(in_channels, in_channels, n_attributes)
        self.crt = CRT(in_channels, in_channels//4, n_attributes)
        self.rim = RIM(in_channels, n_attributes) 

    def forward(self, x):
        idnt, attr = self.cfc(x)

        attr_crt = self.crt(attr)
        idnt = self.rim(idnt, attr_crt)

        return idnt, attr