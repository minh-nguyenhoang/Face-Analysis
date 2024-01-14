import torch
import torch.nn as nn
from .focal_loss import FocalLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

class BinFocalLoss(nn.Module):
    
    def __init__(self, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )
class multi_task_loss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.age_loss = FocalLoss(gamma= 2, alpha=torch.tensor([1.,1.,1.,0.25,1.,1.]).to(device))
        self.gender_loss = BinFocalLoss(weight=torch.tensor([0.5, 1.]).to(device))
        self.masked_loss = nn.BCEWithLogitsLoss()
        self.race_loss = FocalLoss(gamma= 2, alpha=torch.tensor([0.25, 0.25, 1.]).to(device))
        self.skin_loss = FocalLoss(gamma= 2, alpha=torch.tensor([ 0.25, 0.5, 1., 1.]).to(device))
        self.emo_loss = FocalLoss(gamma= 2, alpha=torch.tensor([0.25, 0.25, 1.,1.,1.,1.,1.]).to(device))

    
    def forward(self, x, age, gender, masked, emotion, race, skin):
        age_pred, race_pred, gender_pred, mask_pred, emotion_pred, skintone_pred = x
        loss_age = self.age_loss(age_pred, age)
        loss_gender = self.gender_loss(gender_pred, gender.unsqueeze(1).float())
        loss_masked = self.masked_loss(mask_pred, masked.unsqueeze(1).float())
        loss_race = self.race_loss(race_pred, race)
        loss_skin = self.skin_loss(skintone_pred, skin)
        loss_emo = self.emo_loss(emotion_pred, emotion)
        return loss_age + loss_gender + loss_masked + loss_emo + loss_race + loss_skin