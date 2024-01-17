import torch
from torch import Tensor
import torch.nn as nn

import torch.nn.functional as F
from ..utils.layers.icn import ICN, CFC
from ..utils.layers import CORAL
import timm
from .vit import ViT_Minh

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

        self.cfc = CFC(in_channels, in_channels, n_attributes= n_attributes)

        # num_patches, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0
        self.vit_head = ViT_Minh(num_patches=7*7, dim=in_channels, depth=4, heads=8, pool='cls', dim_head=128, dropout=0., emb_dropout=0.2)

        self.age_branch = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 6)
        )

        self.race_branch = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 3)
        )

        self.gender_branch = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )
        
        self.masked_branch = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )

        self.emotion_branch = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 7)
        )

        self.skintone_branch = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        feat= self.vcn(x)


        _, attr = self.cfc(feat)

        attr = list(attr.values())
        attr_ = torch.stack(attr, dim= 1)

        b, c, h, w = feat.shape 
        # print(feat.shape)
        feat = feat.view(b, c, -1).permute(0,2,1)
        age, race, gender, mask, emotion, skintone = self.vit_head(feat, attr_)

        age = self.age_branch(age)
        race = self.race_branch(race)
        gender = self.gender_branch(gender)
        mask = self.masked_branch(mask)
        emotion = self.emotion_branch(emotion)
        skintone = self.skintone_branch(skintone)

        age_ = self.age_branch(attr[0])
        race_ = self.race_branch(attr[1])
        gender_ = self.gender_branch(attr[2])
        mask_ = self.masked_branch(attr[3])
        emotion_ = self.emotion_branch(attr[4])
        skintone_ = self.skintone_branch(attr[5])


        if self.training:
            return age, race, gender, mask, emotion, skintone, age_, race_, gender_, mask_, emotion_, skintone_
        else:
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