import torch
from torch import Tensor
import torch.nn as nn

import torch.nn.functional as F
from ..utils.layers.icn import ICN, CFC
from ..utils.layers.groupface import GroupFace_S, GroupFace_M, CDN
from ..utils.layers import CORAL
import timm

# print(timm.list_models(pretrained=True))
class BioNet_S(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels, out_channels:int = 512, n_groups:int = 16, n_attributes:int = 6, fine_tune=False) -> None:
        super().__init__()
        
        if fine_tune:
            for name, param in backbone.named_parameters():
                if "channel_expansion" not in name:
                    param.requires_grad = False
        self.vcn = backbone
        self.in_channels = in_channels

        self.cfc = CFC(in_channels, out_channels, n_attributes)

        self.group_face = GroupFace_S(out_channels, out_channels, groups= n_groups)
        self.contribution_net = CDN(out_channels, n_groups)

        # self.age_branch = CORAL(in_features=out_channels, out_features=6)
        self.age_branch = nn.Sequential(
            nn.Linear(out_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 6)
        )

        self.race_branch = nn.Sequential(
            nn.Linear(out_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 3)
        )
        self.gender_branch = nn.Sequential(
            nn.Linear(out_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1)
        )
        self.masked_branch = nn.Sequential(
            nn.Linear(out_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1)
        )
        self.emotion_branch = nn.Sequential(
            nn.Linear(out_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 7)
        )
        self.skintone_branch = nn.Sequential(
            nn.Linear(out_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        feat = self.vcn(x)

        id, attr = self.cfc(feat)

        add_attr, _, group_prob = self.group_face(id)

        _, contribution = self.contribution_net(add_attr)
        contribution = torch.split(contribution, 1, dim= -1)

        age = self.age_branch(attr['attr_0'] + contribution[0]*add_attr)
        race = self.race_branch(attr['attr_1'] + contribution[1]*add_attr)
        gender = self.gender_branch(attr['attr_2'] + contribution[2]*add_attr)
        mask = self.masked_branch(attr['attr_3'] + contribution[3]*add_attr)
        emotion = self.emotion_branch(attr['attr_4'] + contribution[4]*add_attr)
        skintone = self.skintone_branch(attr['attr_5'] + contribution[5]*add_attr)

        if self.training:
            return age, race, gender, mask, emotion, skintone, group_prob
        else:
            return age, race, gender, mask, emotion, skintone
    
    @classmethod
    def from_inputs(cls, backbone: nn.Module, out_channels:int = 512, n_groups:int = 16, n_attributes:int = 4, inputs: torch.Tensor= None, input_shape: torch.Size=None, fine_tune=False):
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



class BioNet_M(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels, out_channels:int = 512, n_groups:int = 16, n_attributes:int = 6, fine_tune=False) -> None:
        super().__init__()
        
        if fine_tune:
            for name, param in backbone.named_parameters():
                if "channel_expansion" not in name:
                    param.requires_grad = False
        self.vcn = backbone
        self.in_channels = in_channels

        self.cfc = CFC(in_channels, in_channels, n_attributes)

        self.group_face = GroupFace_M(in_channels, out_channels, groups= n_groups, n_attributes=n_attributes)

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

        _, attr = self.cfc(feat)

        attr, _, group_prob = self.group_face(attr.values())

        attr = {f'attr_{i}': v for i,v in enumerate(attr)}

        age = self.age_branch(attr['attr_0'])
        race = self.race_branch(attr['attr_1'])
        gender = self.gender_branch(attr['attr_2'])
        mask = self.masked_branch(attr['attr_3'])
        emotion = self.emotion_branch(attr['attr_4'])
        skintone = self.skintone_branch(attr['attr_5'])
        if self.training:
            return age, race, gender, mask, emotion, skintone, group_prob
        else:
            return age, race, gender, mask, emotion, skintone
    
    @classmethod
    def from_inputs(cls, backbone: nn.Module, out_channels:int = 512, n_groups:int = 16, n_attributes:int = 4, inputs: torch.Tensor= None, input_shape: torch.Size=None, fine_tune=False):
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
    
BioNet = BioNet_S