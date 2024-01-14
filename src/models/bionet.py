import torch
from torch import Tensor
import torch.nn as nn

import torch.nn.functional as F
from ..utils.layers.icn import ICN, CFC
from ..utils.layers.groupface import GroupFace
from ..utils.layers import CORAL
import timm

# print(timm.list_models(pretrained=True))
class BioNet(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels, out_channels:int = 512, n_groups: int = 16, n_attributes:int = 6, fine_tune=False) -> None:
        super().__init__()
        
        if fine_tune:
            for name, param in backbone.named_parameters():
                if "fc" not in name or "features" not in name:
                    param.requires_grad = False
        self.vcn = backbone
        self.in_channels = in_channels

        self.cfc = GroupFace(in_channels, out_channels, n_groups, n_attributes= n_attributes)

        # self.age_branch = CORAL(in_features=out_channels, out_features=6)
        self.age_branch = nn.Sequential(
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )

        self.race_branch = nn.Sequential(
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        self.gender_branch = nn.Sequential(
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.masked_branch = nn.Sequential(
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.emotion_branch = nn.Sequential(
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 7)
        )
        self.skintone_branch = nn.Sequential(
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        feat = self.vcn(x)

        final, group_inter, group_prob = self.cfc(feat)

        age = self.age_branch(final)
        race = self.race_branch(final)
        gender = self.gender_branch(final)
        mask = self.masked_branch(final)
        emotion = self.emotion_branch(final)
        skintone = self.skintone_branch(final)

        if self.training:
            return age, race, gender, mask, emotion, skintone, group_prob
        else:
            return age, race, gender, mask, emotion, skintone
    
    @classmethod
    def from_inputs(cls, backbone: nn.Module, out_channels:int = 512, n_groups: int = 16, n_attributes:int = 6, inputs: torch.Tensor= None, input_shape: torch.Size=None, fine_tune=False):
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
        return cls(backbone, out_shape[1], out_channels, n_groups, n_attributes, fine_tune)
    
    # if __name__ == "__main__":
    #     x = torch.randn((2,3,224,224))
    #     backbone = timm.create_model('convnext_base.clip_laiona_augreg_ft_in1k_384', pretrained=True)
    #     backbone.head = nn.Identity()
    #     model = BioNet(backbone=backbone, in_channels=1024)
    #     print(model(x)[0].shape)