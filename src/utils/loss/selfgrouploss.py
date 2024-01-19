"""
Pytorch implementation of GroupFace: https://arxiv.org/abs/2005.10497
"""


import torch
from torch import Tensor
import torch.nn as nn


class SelfGroupingLoss(nn.Module):
    def __init__(self, weight:Tensor|None = None, size_average = None, ignore_index = -100, reduction = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.critetion = nn.NLLLoss(weight= weight, size_average= size_average, ignore_index= ignore_index, reduction = reduction)

    def forward(self, inputs: Tensor, targets: Tensor= None):

        if  (1. - inputs.sum(-1).mean()).abs() >1e-3:
            inputs = torch.softmax(inputs, sim = -1)

        if targets is None:
            groups =  inputs.shape[-1]
            group_label_E: torch.Tensor = inputs.mean(dim=0, keepdim= True)
            group_label_u: torch.Tensor = (inputs - group_label_E) / groups + 1 / groups
            targets = torch.argmax(group_label_u, dim = -1).detach()

        log_prob = torch.log(inputs + 1e-4)

        loss = self.critetion(log_prob, targets)

        return loss
    
class EntropyMinimization(nn.Module):
    def __init__(self, weight:Tensor|None = None, size_average = None, ignore_index = -100, reduction = 'mean') -> None:
        super().__init__()
        assert reduction in [None, 'none', 'mean', "sum"]
        self.reduction = reduction
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, inputs):
        entropy = - inputs*torch.log(inputs)

        if self.reduction is None or self.reduction == 'none':
            return entropy
        else:
            ops = getattr(torch, self.reduction)
            return ops(entropy)
        