import torch
import torch.nn as nn
from .focal_loss import FocalLoss

class multi_task_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.age_loss = nn.BCEWithLogitsLoss()
        self.gender_loss = nn.BCEWithLogitsLoss()
        self.masked_loss = nn.BCEWithLogitsLoss()
        self.race_loss = FocalLoss()
        self.skin_loss = FocalLoss()
        self.emo_loss = FocalLoss()

    
    def forward(self, x, age, gender, masked, emotion, race, skin):
        age_pred, race_pred, gender_pred, mask_pred, emotion_pred, skintone_pred = x
        loss_age = self.age_loss(age_pred, age)
        loss_gender = self.gender_loss(gender_pred, gender.unsqueeze(1).float())
        loss_masked = self.masked_loss(mask_pred, masked.unsqueeze(1).float())
        loss_race = self.race_loss(race_pred, race)
        loss_skin = self.skin_loss(skintone_pred, skin)
        loss_emo = self.emo_loss(emotion_pred, emotion)
        return loss_age + loss_gender + loss_masked + loss_emo + loss_race + loss_skin