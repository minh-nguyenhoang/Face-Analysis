import torch
from torch import Tensor
import torch.nn as nn

import torch.nn.functional as F
from ..utils.layers.icn import ICN, CFC
from ..utils.layers import CORAL
import timm

# print(timm.list_models(pretrained=True))
class BioNet(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels, out_channels:int = 512, n_attributes:int = 6) -> None:
        super().__init__()
        self.vcn = backbone

        self.in_channels = in_channels

        self.cfc = CFC(in_channels, out_channels, n_attributes)

        self.age_branch = CORAL(in_features=out_channels, out_features=5)

        self.race_branch = nn.Sequential(
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 3)
        )
        self.gender_branch = nn.Sequential(
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1)
        )
        self.masked_branch = nn.Sequential(
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1)
        )
        self.emotion_branch = nn.Sequential(
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 5)
        )
        self.skintone_branch = nn.Sequential(
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        feat = self.vcn(x)

        _, attr = self.cfc(feat)

        age = self.age_branch(attr['attr_0'])
        race = self.race_branch(attr['attr_1'])
        gender = self.gender_branch(attr['attr_2'])
        mask = self.masked_branch(attr['attr_3'])
        emotion = self.emotion_branch(attr['attr_4'])
        skintone = self.skintone_branch(attr['attr_5'])

        return age, race, gender, mask, emotion, skintone
    
    @classmethod
    def from_inputs(cls, backbone: nn.Module, out_channels:int = 512, n_attributes:int = 4, inputs: torch.Tensor= None, input_shape: torch.Size=None,):
        if inputs is None:
            assert input_shape is not None, "Input shape must not be empty if inputs is not provided."
            device = next(backbone.parameters()).device
            inputs = torch.rand(input_shape).to(device)

        train_state = backbone.training

        backbone.eval()
        out_shape = backbone(inputs).shape
        backbone.train(train_state)
        return cls(backbone, out_shape[1], out_channels, n_attributes)
    
    # if __name__ == "__main__":
    #     x = torch.randn((2,3,224,224))
    #     backbone = timm.create_model('convnext_base.clip_laiona_augreg_ft_in1k_384', pretrained=True)
    #     backbone.head = nn.Identity()
    #     model = BioNet(backbone=backbone, in_channels=1024)
    #     print(model(x)[0].shape)