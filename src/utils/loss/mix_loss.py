import torch
import torch.nn as nn

class multi_task_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.age_loss = nn.BCELoss()
        self.gender_loss = nn.BCELoss()
        self.masked_loss = nn.BCELoss()
        self.race_loss = nn.CrossEntropyLoss()
        self.skin_loss = nn.CrossEntropyLoss()
        self.emo_loss = nn.CrossEntropyLoss()

    
    def forward(self, x, age, gender, masked, emotion, race, skin):
        age_pred, race_pred, gender_pred, mask_pred, emotion_pred, skintone_pred = x
        loss_1 = self.age_loss(age_pred, age)
        loss_2 = self.gender_loss(gender_pred, gender)
        loss_3 = self.masked_loss(mask_pred, masked)
        loss_4 = self.race_loss(race_pred, race)
        loss_5 = self.skin_loss(skintone_pred, skin)
        loss_6 = self.emo_loss(emotion_pred, emotion)
        return torch.mean(loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6)