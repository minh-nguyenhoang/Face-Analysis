import torch
import torch.nn as nn
from .focal_loss import FocalLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

class BinFocalLoss(nn.Module):
    
    def __init__(self, alpha=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        alpha_tensor = (1 - self.alpha) + targets * (2 * self.alpha - 1)  # alpha if target = 1 and 1 - alpha if target = 0
        f_loss = alpha_tensor * (1 - p_t) ** self.gamma * bce_loss
        return f_loss.mean()
    
class multi_task_loss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.age_loss = FocalLoss(gamma= 2, alpha=torch.tensor([1.,1.,1.,0.25,1.,1.]).to(device))
        self.gender_loss = BinFocalLoss(alpha=torch.tensor(0.25).to(device))
        self.masked_loss =  BinFocalLoss(alpha=torch.tensor(0.25).to(device))
        self.race_loss = FocalLoss(gamma= 2, alpha=torch.tensor([0.25, 0.25, 1.]).to(device))
        self.skin_loss = FocalLoss(gamma= 2, alpha=torch.tensor([ 0.25, 0.5, 1., 1.]).to(device))
        self.emo_loss = FocalLoss(gamma= 2, alpha=torch.tensor([0.25, 0.25, 1.,1.,1.,1.,1.]).to(device))

    
    def forward(self, x, age, gender, masked, emotion, race, skin):
        if len(x) == 6:
            age_pred, race_pred, gender_pred, mask_pred, emotion_pred, skintone_pred = x
        else:
            age_pred, race_pred, gender_pred, mask_pred, emotion_pred, skintone_pred, age_pred_, race_pred_, gender_pred_, mask_pred_, emotion_pred_, skintone_pred_ = x
        loss_age = self.age_loss(age_pred, age)
        loss_gender = self.gender_loss(gender_pred, gender.unsqueeze(1).float())
        loss_masked = self.masked_loss(mask_pred, masked.unsqueeze(1).float())
        loss_race = self.race_loss(race_pred, race)
        loss_skin = self.skin_loss(skintone_pred, skin)
        loss_emo = self.emo_loss(emotion_pred, emotion)

        if len(x) != 6:
            loss_age_ = self.age_loss(age_pred_, age)
            loss_gender_ = self.gender_loss(gender_pred_, gender.unsqueeze(1).float())
            loss_masked_ = self.masked_loss(mask_pred_, masked.unsqueeze(1).float())
            loss_race_ = self.race_loss(race_pred_, race)
            loss_skin_ = self.skin_loss(skintone_pred_, skin)
            loss_emo_ = self.emo_loss(emotion_pred_, emotion)

            return (loss_age + loss_gender + loss_masked + loss_emo + loss_race + loss_skin) * 0.85 + (loss_age_ + loss_gender_ + loss_masked_ + loss_emo_ + loss_race_ + loss_skin_) *0.15
        return loss_age + loss_gender + loss_masked + loss_emo + loss_race + loss_skin