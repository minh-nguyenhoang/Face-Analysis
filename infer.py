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
from src.models.iresnet import iresnet100
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

        return scaled_padded_image, tl, scale, image


    
    def __getitem__(self, index) -> Any:
        path = os.path.join(self.root, self.file_names[index])

        image, tl, scale, raw_image = self.get_image(path)

        return torch.tensor(image), torch.tensor(tl), torch.tensor(scale), self.position[index], self.file_names[index], raw_image
            

def collate_fn(batch):
    '''
    return pseudo batch with uneven
    '''

    ret = []

    for item in zip(*batch):
        if isinstance(item[0], torch.Tensor):
            try:
                ret.append(torch.stack(item, dim= 0))
            except:
                ret.append(item)

    return ret


@torch.no_grad()
def main():
    '''
    Assume that the loader has batch size of 1 due to each uncrop image has different size
    '''

    args = parser.parse_args()

    device = 'cuda'
    bbox_expand_scale = 0.

    face_detector = RetinaFace(network= 'resnet50', device= device, gpu_id= None)

    # backbone: nn.Module = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=False)
    # backbone.head = nn.Identity()

    backbone = iresnet100(pretrained= True, num_features = 4096,)

    model: BioNet = BioNet.from_inputs(backbone= backbone, out_channels= 512, n_attributes= 6, input_shape=(1,3,112,112))
    try:
        model.load_state_dict(torch.load(f'/kaggle/input/baseline-checkpoint/{args.fname}', map_location= 'cpu'))
    except:
        model.load_state_dict(torch.load(f'{args.fname}', map_location= 'cpu'))

    model = model.to(face_detector.device)
    model.eval()

    test_dataset = TestDataset(root= '/kaggle/input/pixte-public-test/public_test/public_test', json_file= '/kaggle/input/pixte-public-test/public_test_and_submission_guidelin/public_test_and_submission_guidelines/file_name_to_image_id.json')
    test_dataloader: DataLoader = DataLoader(test_dataset, batch_size= 16, collate_fn= collate_fn)

    file_names, bboxes, image_id, ages, races, genders, masks, emotions, skintones = [], [], [], [], [], [], [], [], []

    for idx, batch in enumerate(tqdm(test_dataloader)):
        '''
        tl should have size [Bx2]
        scale should have size [B]
        '''
        images, tl, scale, position, file_path, raw_images = batch
        images = images.to(device) 

        tl = tl.view(-1,2) #[w,h]
        tl = torch.cat([tl,tl], dim = 1) # [Bx4]
        scale = scale.view(-1, 1)

        '''
        Only take the first bbox found for each image (if any).
        Checked in public_test, each image only has 1 bbox. 
        '''
        dets = face_detector(images) # [Bx4]
        images_, corners, tl_, scale_, position_, file_path_, raw_images_ = [], [], [], [], [], [], []
        
        # corners = [[min(max(tl[idx][0], det[0][0][0] - (det[0][0][2] -det[0][0][0])* bbox_expand_scale/2), 1024- tl[idx][0] + (det[0][0][2] -det[0][0][0])* bbox_expand_scale/2), 
        #             min(max(tl[idx][1], det[0][0][1] - (det[0][0][3] -det[0][0][1])* bbox_expand_scale/2), 1024- tl[idx][1] + (det[0][0][3] -det[0][0][1])* bbox_expand_scale/2), 
        #             min(max(tl[idx][0], det[0][0][2] - (det[0][0][2] -det[0][0][0])* bbox_expand_scale/2), 1024- tl[idx][0] + (det[0][0][2] -det[0][0][0])* bbox_expand_scale/2), 
        #             min(max(tl[idx][1], det[0][0][3] - (det[0][0][3] -det[0][0][1])* bbox_expand_scale/2), 1024- tl[idx][1] + (det[0][0][3] -det[0][0][1])* bbox_expand_scale/2)] 
        #         if len(det) >0 else [tl[idx][0], tl[idx][1], 1024 - tl[idx][0], 1024 - tl[idx][1]] for idx, det in enumerate(dets)]
        
        for idx, det in enumerate(dets):
            if len(det) >0:
                for coord in det:
                    images_.append(images[idx].clone())
                    corners.append([min(max(tl[idx][0].numpy(), coord[0][0] - (coord[0][2] - coord[0][0]) * bbox_expand_scale/2), 1024 - tl[idx][0].numpy() + (coord[0][2] - coord[0][0]) * bbox_expand_scale/2), 
                                    min(max(tl[idx][1].numpy(), coord[0][1] - (coord[0][3] - coord[0][1]) * bbox_expand_scale/2), 1024 - tl[idx][1].numpy() + (coord[0][3] - coord[0][1]) * bbox_expand_scale/2), 
                                    min(max(tl[idx][0].numpy(), coord[0][2] - (coord[0][2] - coord[0][0]) * bbox_expand_scale/2), 1024 - tl[idx][0].numpy() + (coord[0][2] - coord[0][0]) * bbox_expand_scale/2), 
                                    min(max(tl[idx][1].numpy(), coord[0][3] - (coord[0][3] - coord[0][1]) * bbox_expand_scale/2), 1024 - tl[idx][1].numpy() + (coord[0][3] - coord[0][1]) * bbox_expand_scale/2)])
                    tl_.append(tl[idx])
                    scale_.append(scale[idx])
                    position_.append(position[idx])
                    file_path_.append(file_path[idx])
                    raw_images_.append(raw_images[idx])
            else:
                images_.append(images[idx].clone())
                corners.append([tl[idx][0], tl[idx][1], 1024 - tl[idx][0], 1024 - tl[idx][1]])
                tl_.append(tl[idx])
                scale_.append(scale[idx])
                position_.append(position[idx])
                file_path_.append(file_path[idx])
                raw_images_.append(raw_images[idx])



        xyxy = torch.tensor(np.array(corners)).sub(torch.stack(tl_)).div(torch.tensor(scale_).view(-1,1)).int() #[Bx4] 

        tl_x, tl_y, br_x, br_y = zip(*xyxy)
        xywh = torch.stack([torch.tensor(tl_x), torch.tensor(tl_y), torch.tensor(br_x) - torch.tensor(tl_x), torch.tensor(br_y) - torch.tensor(tl_y)], dim = -1)



        bboxes.extend(xywh.tolist())
        image_id.extend(position_)
        file_names.extend(file_path_)

        images_ = torch.tensor(
            np.array([letterbox(image[tl_x:br_x, tl_y:br_y].cpu().numpy(), (112,112)) for image, corner in zip(raw_images_, corners)])
            ).to(device)
        images_ = images_.permute(0,3,1,2).div(255.0).sub(torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1).to(device)).div(torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1).to(device))


        age, race, gender, mask, emotion, skintone = model(images_)

        # age: torch.Tensor = torch.sum(age.sigmoid() > 0.5, dim =1)
        age = torch.argmax(age, dim = 1)
        race = torch.argmax(race, dim = 1)

        gender = torch.squeeze(torch.sigmoid(gender)  > 0.5, dim = 1).int()
        mask = torch.squeeze(torch.sigmoid(mask)  > 0.5, dim = 1).int()

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
    submission_file['file_name'] = pd.Series(file_names, dtype= str)
    submission_file['bbox'] = pd.Series(bboxes)
    submission_file['image_id'] = pd.Series(image_id, dtype= str)
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









