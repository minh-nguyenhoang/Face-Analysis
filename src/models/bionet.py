import torch
from torch import Tensor
import torch.nn as nn

import torch.nn.functional as F
from ..utils.layers.icn import ICN, CFC
from ..utils.layers import CORAL
import timm
from .vit import ViT_Dung

# print(timm.list_models(pretrained=True))
class BioNet(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels, out_channels:int = 512, n_attributes:int = 6, fine_tune=False) -> None:
        super().__init__()
        
        if fine_tune:
            for name, param in backbone.named_parameters():
                if "channel_expansion" not in name:
                    param.requires_grad = False
        self.vcn = backbone
        self.in_channels = in_channels

        # num_patches, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0
        self.vit_head = ViT_Dung(num_patches=7*7, dim=in_channels, depth=6, heads=8, mlp_dim=out_channels, pool='cls', dim_head=128, dropout=0.2, emb_dropout=0.2)

    def forward(self, x):
        feat = self.vcn(x)
        b, c, h, w = feat.shape 
        # print(feat.shape)
        feat = feat.view(b, c, -1).permute(0,2,1)
        age, race, gender, mask, emotion, skintone = self.vit_head(feat)

        return age, race, gender, mask, emotion, skintone
    
    @classmethod
    def from_inputs(cls, backbone: nn.Module, out_channels:int = 512, n_attributes:int = 4, inputs: torch.Tensor= None, input_shape: torch.Size=None, fine_tune=False):
        if inputs is None:
            assert input_shape is not None, "Input shape must not be empty if inputs is not provided."
            device = next(backbone.parameters()).device
            inputs = torch.rand(input_shape).to(device)

        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)

        train_state = backbone.training

        backbone.eval()
        out_shape = backbone(inputs).shape
        backbone.train(train_state)
        return cls(backbone, out_shape[1], out_channels, n_attributes, fine_tune)
    
    # if __name__ == "__main__":
    #     x = torch.randn((2,3,224,224))
    #     backbone = timm.create_model('convnext_base.clip_laiona_augreg_ft_in1k_384', pretrained=True)
    #     backbone.head = nn.Identity()
    #     model = BioNet(backbone=backbone, in_channels=1024)
    #     print(model(x)[0].shape)