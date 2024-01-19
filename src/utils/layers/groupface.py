"""
Pytorch implementation of GroupFace: https://arxiv.org/abs/2005.10497
"""


import torch
import torch.nn as nn
from .frelu import FReLU
from .gdn import GDN, FC

class GroupFace(nn.Module):
    def __init__(self, in_channels=768, out_channels=512, groups=32, mode='S'):
        super(GroupFace, self).__init__()
        self.mode = mode
        self.groups = groups
        self.instance_fc = FC(in_channels, out_channels)
        self.gdn = GDN(out_channels, groups)
        self.group_fc = nn.ModuleList([FC(in_channels, out_channels) for i in range(groups)])
        
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor):
        instacne_representation = self.instance_fc(x)
        # GDN
        group_inter, group_prob: torch.Tensor = self.gdn(instacne_representation)
        # group aware repr

        # group ensemble
        if self.mode == 'S':
            v_G = torch.stack([Gk(x) for Gk in self.group_fc], dim= 1)  # (B,n_groups,512)
            group_ensembled = torch.mul(v_G, group_prob).sum(dim=1)
        else:
            label = torch.argmax(group_prob, dim=1)
            group_ensembled = self.group_fc[label](x)    
        
        final = instacne_representation + group_ensembled
        return final, group_inter, group_prob