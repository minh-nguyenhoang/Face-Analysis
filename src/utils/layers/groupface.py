"""
Pytorch implementation of GroupFace: https://arxiv.org/abs/2005.10497
"""


import torch
import torch.nn as nn
from .frelu import FReLU
from .gdn import GDN, FC

class GroupFace(nn.Module):
    def __init__(self, in_channels=1024, out_channels=256, groups=4, mode='S', n_attributes:int = 6):
        super(GroupFace, self).__init__()
        self.mode = mode
        self.groups = groups
        self.n_attributes = n_attributes
        self.instance_fc = FC(in_channels, out_channels)
        self.gdn = GDN(out_channels, groups)
        self.group_fc = nn.ModuleList([FC(in_channels, out_channels) for i in range(groups)])
        
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor):
        instacne_representation = self.instance_fc(x)
        # GDN
        group_inter, group_prob = self.gdn(instacne_representation)
        # group aware repr

        # group ensemble
        if self.mode == 'S':
            v_G = torch.stack([Gk(x) for Gk in self.group_fc], dim= 1)  # (B,n_groups,512)
            group_ensembled = torch.mul(v_G, group_prob.view(group_prob.shape[0], group_prob.shape[1], 1)).sum(dim=1)
        else:
            label = torch.argmax(group_prob, dim=1)
            group_ensembled = self.group_fc[label](x)    
        
        final = instacne_representation + group_ensembled
        return [final]*self.n_attributes, group_inter, group_prob
    

# class GroupFace(nn.Module):
#     def __init__(self, in_channels=1024, out_channels=256, groups=4, mode='S', n_attributes:int = 6):
#         super(GroupFace, self).__init__()
#         self.mode = mode
#         self.groups = groups
#         self.instance_fc = nn.ModuleList([nn.Sequential(
#                                             FC(in_channels, in_channels// 8),
#                                             nn.Linear(in_channels// 8, out_channels)
#         ) for i in range(n_attributes)])
#         self.gdn = GDN(out_channels*n_attributes, groups)
#         self.group_fc = nn.ModuleList([nn.Sequential(
#                                             FC(in_channels, in_channels// 8),
#                                             nn.Linear(in_channels// 8, out_channels)
#         ) for i in range(groups)])
        
#         self.out_channels = out_channels
#         self.n_attributes = n_attributes

#         self.attribute_disentangle = nn.Linear(out_channels, n_attributes, bias= False)


#     def forward(self, x: torch.Tensor):
#         instacne_representation = [module(x) for module in self.instance_fc] 
#         # GDN
#         group_inter, group_prob = self.gdn(torch.cat(instacne_representation,dim= -1))
#         # group aware repr

#         # group ensemble
#         if self.mode == 'S':
#             v_G = torch.stack([Gk(x) for Gk in self.group_fc], dim= 1)  # (B,n_groups,512)
#             group_ensembled = torch.mul(v_G, group_prob.view(group_prob.shape[0], group_prob.shape[1], 1)).sum(dim=1)
#         else:
#             label = torch.argmax(group_prob, dim=1)
#             group_ensembled = self.group_fc[label](x)    

#         coeffs = torch.split(self.attribute_disentangle(group_ensembled), 1, dim= 1)
#         final:torch.Tensor = [attr_rep + coeff*group_ensembled for attr_rep, coeff in zip(instacne_representation, coeffs)]
#         return final, group_inter, group_prob