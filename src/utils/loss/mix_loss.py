import torch
import torch.nn as nn
from .focal_loss import FocalLoss

class Bin_FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2): 
        super(Bin_FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.binary_ce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, inputs, targets): 
        BCE_loss = self.binary_ce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)
    
class multi_task_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.age_loss = FocalLoss(gamma= 2)
        self.gender_loss = Bin_FocalLoss()
        self.masked_loss = Bin_FocalLoss()
        self.race_loss = FocalLoss(gamma= 2)
        self.skin_loss = FocalLoss(gamma= 2)
        self.emo_loss = FocalLoss(gamma= 2)

    
    def forward(self, x, age, gender, masked, emotion, race, skin):
        age_pred, race_pred, gender_pred, mask_pred, emotion_pred, skintone_pred = x
        loss_age = self.age_loss(age_pred, age)
        loss_gender = self.gender_loss(gender_pred, gender.unsqueeze(1).float())
        loss_masked = self.masked_loss(mask_pred, masked.unsqueeze(1).float())
        loss_race = self.race_loss(race_pred, race)
        loss_skin = self.skin_loss(skintone_pred, skin)
        loss_emo = self.emo_loss(emotion_pred, emotion)
        return loss_age + loss_gender + loss_masked + loss_emo + loss_race + loss_skin