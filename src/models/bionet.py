import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from ..utils.layers.icn import ICN

class BioNet(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels, out_channels:int = 512, n_attributes:int = 4) -> None:
        super().__init__()
        self.vcn = backbone

        in_channels = ...

        self.icn = ICN(in_channels, out_channels, n_attributes)

    def forward(self, x):
        feat = self.vcn(x)

        idnt, attr = self.icn(feat)

        return idnt, attr 
    
    @classmethod
    def from_inputs(cls, backbone: nn.Module, out_channels:int = 512, n_attributes:int = 4, inputs: torch.Tensor|None= None, input_shape: torch.Size|None = None,):
        if inputs is None:
            assert input_shape is not None, "Input shape must not be empty if inputs is not provided."
            device = next(backbone.parameters()).device
            inputs = torch.rand(input_shape).to(device)

        train_state = backbone.training

        backbone.eval()
        out_shape = backbone(inputs).shape
        backbone.train(train_state)
        return cls(backbone, out_shape[1], out_channels, n_attributes)