from turtle import back
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F

import torchvision.models.detection.backbone_utils as backbone_utils
from collections import OrderedDict

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from src.models import BioNet
import numpy as np
import pandas as pd
import os
import cv2
import time
import json
from typing import Dict, Any
from src.utils import RetinaFace
from src.utils.data_process.letterbox import letterbox
from src.utils.label_mapping import LabelMapping
import timm


class TestDataset(Dataset):
    def __init__(self, 
                 root,
                 json_file = 'file_name_to_image_id.json',
                 temp_size = (1024, 1024)) -> None:
        super().__init__()

        try: 
            file: Dict = json.load(open(json_file))
        except:
            try:
                file: Dict = json.load(open(os.path.join(root, json_file)))
            except Exception as e:
                raise e
            
        file_names, position = file.items()

        self.root = root
        self.file_names = file_names[(np.array(position)-1).tolist]
        self.position = position
        

        self.temp_size = temp_size

    def __len__(self):
        return len(self.file_names)
    
    def get_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        assert image is not None

        scaled_padded_image, tl, scale = letterbox(image, self.temp_size, return_extra_args= True)

        return scaled_padded_image, tl, scale


    
    def __getitem__(self, index) -> Any:
        path = os.path.join(self.root, self.file_names[index])

        image, tl, scale = self.get_image(path)

        return torch.tensor(image), torch.tensor(tl), torch.tensor(scale)
            

def collate_fn(batch):
    pass


def main(args):
    '''
    Assume that the loader has batch size of 1 due to each uncrop image has different size
    '''

    device = 'cuda'

    face_detector = RetinaFace(network= 'mobilenet', device= device, gpu_id= None)

    backbone: nn.Module = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)
    backbone.head = nn.Identity()

    model = BioNet(backbone, 1024, 512)
    model.to(device)
    net: BioNet = BioNet.from_inputs(backbone= backbone, out_channels= 512, n_attributes= 6, input_shape=(1,3,224,224))
    # net.load_state_dict(...)
    net = net.to(face_detector.device)
    net.eval()

    test_dataloader: DataLoader = ...

    bboxes, ages, races, genders, masks, emotions, skintones = [], [], [], [], [], [], []

    for idx, batch in enumerate(test_dataloader):
        '''
        tl should have size [Bx2]
        scale should have size [B]
        '''
        images, tl, scale = batch
        images = images.to(device) 

        tl = tl.view(-1,2)
        scale = scale.view(-1)

        '''
        Only take the first bbox found for each image (if any).
        Checked in public_test, each image only has 1 bbox. 
        '''
        corners = face_detector(images)[..., 0, 0] # [Bx4]

        bboxes.extend(torch.tensor(corners).sub(tl).div(scale).int().tolist())

        images = torch.tensor(
            [letterbox(image[corner[0]:corner[2], corner[1]: corner[3]].cpu().numpy()) for image, corner in zip(images, corners)]
            ).to(device)
        images = images.permute(0,3,1,2).div(255).sub(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).div(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))


        with torch.no_grad():
            age, race, gender, mask, emotion, skintone = net(images)

            age = torch.sum(age > 0.5, dim =1)
            race = torch.argmax(race, dim = 1)
            gender = torch.argmax(gender, dim = 1)
            mask = torch.argmax(mask, dim = 1)
            emotion = torch.argmax(emotion, dim = 1)
            skintone = torch.argmax(skintone, dim = 1)

            age_pred = LabelMapping.age_map_rev[age]
            race_pred = LabelMapping.race_map_rev[race]
            gender_pred = LabelMapping.gender_map_rev[gender]
            mask_pred = LabelMapping.masked_map_rev[mask]
            emotion_pred = LabelMapping.emotion_map_rev[emotion]
            skintone_pred = LabelMapping.skintone_map_rev[skintone]

            ages.extend(age_pred.tolist())
            races.extend(race_pred.tolist())
            genders.extend(gender_pred.tolist())
            masks.extend(mask_pred.tolist())
            emotions.extend(emotion_pred.tolist())
            skintones.extend(skintone_pred.tolist())

    
    submission_file = pd.DataFrame()
    submission_file['file_name'] = pd.Series(test_dataloader.dataset.file_names)
    submission_file['bbox'] = pd.Series(bboxes)
    submission_file['image_id'] = pd.Series(test_dataloader.dataset.position)
    submission_file['race'] = pd.Series(races)
    submission_file['age'] = pd.Series(ages)
    submission_file['emotion'] = pd.Series(emotions)
    submission_file['gender'] = pd.Series(genders)
    submission_file['skintone'] = pd.Series(skintones)
    submission_file['masked'] = pd.Series(masks)

    submission_file.to_csv('answer.csv', sep= ',')


if __name__ == 'main':
    main()









