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
from tqdm.auto import tqdm
from argparse import ArgumentParser

parser = ArgumentParser(
                    prog='Pixta AI Hackathon - Face Analysis',
                    description='Inference file',
                    epilog='Text at the bottom of help')

parser.add_argument('-f' ,'--fname', default= 'best_model.pth')

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
            
        file_names, position = zip(*file.items())

        self.root = root
        self.file_names = file_names
        self.position = position
        

        self.temp_size = temp_size

    def __len__(self):
        return len(self.file_names)
    
    def get_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        assert image is not None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        scaled_padded_image, tl, scale = letterbox(image, self.temp_size, return_extra_args= True)

        return scaled_padded_image, tl, scale


    
    def __getitem__(self, index) -> Any:
        path = os.path.join(self.root, self.file_names[index])

        image, tl, scale = self.get_image(path)

        return torch.tensor(image), torch.tensor(tl), torch.tensor(scale)
            

def collate_fn(batch):
    '''
    return pseudo batch with uneven
    '''
    return zip(*batch)


@torch.no_grad()
def main():
    '''
    Assume that the loader has batch size of 1 due to each uncrop image has different size
    '''

    args = parser.parse_args()

    device = 'cuda'
    bbox_expand_scale = 0.1 

    face_detector = RetinaFace(network= 'resnet50', device= device, gpu_id= None)

    backbone: nn.Module = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=False)
    backbone.head = nn.Identity()

    model = BioNet(backbone, 1024, 512)
    model.to(device)
    net: BioNet = BioNet.from_inputs(backbone= backbone, out_channels= 512, n_attributes= 6, input_shape=(1,3,224,224))
    net.load_state_dict(torch.load(f'/kaggle/input/baseline-checkpoint/{args.fname}', map_location= 'cpu'))
    net = net.to(face_detector.device)
    net.eval()

    test_dataset = TestDataset(root= '/kaggle/input/pixte-public-test/public_test/public_test', json_file= '/kaggle/input/pixte-public-test/public_test_and_submission_guidelin/public_test_and_submission_guidelines/file_name_to_image_id.json')
    test_dataloader: DataLoader = DataLoader(test_dataset, batch_size= 16)

    bboxes, ages, races, genders, masks, emotions, skintones = [], [], [], [], [], [], []

    for idx, batch in enumerate(tqdm(test_dataloader)):
        '''
        tl should have size [Bx2]
        scale should have size [B]
        '''
        images, tl, scale = batch
        images = images.to(device) 

        tl = tl.view(-1,2) #[w,h]
        tl = torch.cat([tl,tl], dim = 1)
        scale = scale.view(-1, 1)

        '''
        Only take the first bbox found for each image (if any).
        Checked in public_test, each image only has 1 bbox. 
        '''
        dets = face_detector(images) # [Bx4]
        corners = []
        corners = [[min(max(tl[idx][0], det[0][0][0] - (det[0][0][2] -det[0][0][0])* bbox_expand_scale/2), 1024- tl[idx][0] + (det[0][0][2] -det[0][0][0])* bbox_expand_scale/2), 
                    min(max(tl[idx][1], det[0][0][1] - (det[0][0][3] -det[0][0][1])* bbox_expand_scale/2), 1024- tl[idx][1] + (det[0][0][3] -det[0][0][1])* bbox_expand_scale/2), 
                    min(max(tl[idx][0], det[0][0][2] - (det[0][0][2] -det[0][0][0])* bbox_expand_scale/2), 1024- tl[idx][0] + (det[0][0][2] -det[0][0][0])* bbox_expand_scale/2), 
                    min(max(tl[idx][1], det[0][0][3] - (det[0][0][3] -det[0][0][1])* bbox_expand_scale/2), 1024- tl[idx][1] + (det[0][0][3] -det[0][0][1])* bbox_expand_scale/2)] 
                if len(det) >0 else [tl[idx][0], tl[idx][1], 1024 - tl[idx][0], 1024 - tl[idx][1]] for idx, det in enumerate(dets)]
        # for idx, det in enumerate(dets):
        #     if len(det) >0:
        #         coord = det[0][0]
        #         corners.append([min(max(tl[idx][0], coord[0]), 1024- tl[idx][0]), min(max(tl[idx][1], coord[1]), 1024- tl[idx][1]), 
        #                         min(max(tl[idx][0], coord[2]), 1024- tl[idx][0]), min(max(tl[idx][1], coord[3]), 1024- tl[idx][1])])
        #     else:
        #         corners.append([tl[idx][0], tl[idx][1], 1024 - tl[idx][0], 1024 - tl[idx][1]])

        xyxy = torch.tensor(np.array(corners)).sub(tl).div(scale).int() #[Bx4]

        tl_x, tl_y, br_x, br_y = zip(*xyxy)
        xywh = torch.stack([torch.tensor(tl_x), torch.tensor(tl_y), torch.tensor(br_x) - torch.tensor(tl_x), torch.tensor(br_y) - torch.tensor(tl_y)], dim = -1)



        bboxes.extend(xywh.tolist())

        images = torch.tensor(
            np.array([letterbox(image[int(corner[0]):int(corner[2]), int(corner[1]): int(corner[3])].cpu().numpy()) for image, corner in zip(images, corners)])
            ).to(device)
        images = images.permute(0,3,1,2).div(255).sub(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)).div(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device))


        age, race, gender, mask, emotion, skintone = net(images)

        # age: torch.Tensor = torch.sum(age.sigmoid() > 0.5, dim =1)
        age = torch.argmax(age, dim = 1)
        race = torch.argmax(race, dim = 1)

        gender = torch.squeeze(torch.sigmoid(gender)  > 0.5, dim = 1)
        mask = torch.squeeze(torch.sigmoid(mask)  > 0.5, dim = 1)

        emotion = torch.argmax(emotion, dim = 1)
        skintone = torch.argmax(skintone, dim = 1)

        age_pred = [LabelMapping.get('age_map_rev').get(a, None) for a in age.cpu().tolist()]
        race_pred = [LabelMapping.get('race_map_rev').get(a, None) for a in race.cpu().tolist()]
        gender_pred = [LabelMapping.get('gender_map_rev').get(a, None) for a in gender.cpu().tolist()]
        mask_pred = [LabelMapping.get('masked_map_rev').get(a, None) for a in mask.cpu().tolist()]
        emotion_pred = [LabelMapping.get('emotion_map_rev').get(a, None) for a in emotion.cpu().tolist()]
        skintone_pred = [LabelMapping.get('skintone_map_rev').get(a, None) for a in skintone.cpu().tolist()]

        ages.extend(age_pred)
        races.extend(race_pred)
        genders.extend(gender_pred)
        masks.extend(mask_pred)
        emotions.extend(emotion_pred)
        skintones.extend(skintone_pred)

    
    submission_file = pd.DataFrame()
    submission_file['file_name'] = pd.Series(test_dataloader.dataset.file_names, dtype= str)
    submission_file['bbox'] = pd.Series(bboxes)
    submission_file['image_id'] = pd.Series(test_dataloader.dataset.position, dtype= str)
    submission_file['race'] = pd.Series(races)
    submission_file['age'] = pd.Series(ages)
    submission_file['age'].fillna(LabelMapping.get('age_map_rev').get(3))
    submission_file['emotion'] = pd.Series(emotions)
    submission_file['gender'] = pd.Series(genders)
    submission_file['skintone'] = pd.Series(skintones)
    submission_file['masked'] = pd.Series(masks)

    # submission_file.set_index('file_name')
    submission_file['age'].fillna('20-30s')

    submission_file.to_csv('answer.csv', sep= ',', index= False)

main()

# if __name__ == 'main':
#     main()









