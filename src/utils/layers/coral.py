import torch
import torch.nn as nn
import torch.nn.functional as F

class CORAL(nn.Module):
    def __init__(self, in_features, out_features, dtype = None, device = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.Parameter(torch.rand(1, in_features, dtype= dtype, device= device), requires_grad= True)
        self.bias = nn.Parameter(torch.rand(out_features, dtype= dtype, device= device), requires_grad= True)

    def forward(self, x: torch.Tensor):
        out = F.linear(x, weight= self.weights, bias= None) + self.bias
        return out
