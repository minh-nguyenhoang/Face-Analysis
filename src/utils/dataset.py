from torch.utils.data import Dataset, DataLoader
import albumentations
import torch
from .data_process.letterbox import letterbox
import pandas as pd
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

class PixtaDataset(Dataset):
    def __init__(self, 
                 root = '.',
                 csv_file = '',
                 img_size = (224,224),
                 phase ='train',
                 transform = None,) -> None:
        super().__init__()
        self.phase = phase
        self.root = root
        self.img_size = img_size

        try:
            self.metadata = pd.read_csv(os.path.join(root, csv_file))
            self.metadata_path = os.path.join(root, csv_file)
        except:
            try:
                self.metadata = pd.read_csv(csv_file)
                self.metadata_path = csv_file
            except Exception as e:
                raise e
            
        # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.train_transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.02, 0.02)],p = 0.5),
                                            transforms.RandomApply([transforms.RandomAffine(5, (0.1,0.1), (1.0,1.25))], p=0.2),
                                            cutout(),
                                            RandomGammaCorrection(),
                                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                                            ])
        self.test_transform = transforms.Compose([
                                        #    transforms.ToTensor(),
                                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                                            ])
        
        self.age_map = {'Baby':0, 'Kid': 1, 'Teenager': 2,'Senior': 4,'20-30s': 3,'40-50s': 5}
        self.race_map = {
            'Mongoloid': 0,
            'Caucasian': 1,
            'Negroid': 2
        }
        self.masked_map = {
            'masked': 0,
            'unmasked':1
        }
        self.skintone_map = {
            'light': 0,
            'mid-light': 1,
            'mid-dark' :2,
            'dark': 3
        }
        self.emotion_map = {
            'Happiness': 0,
            'Neutral':1,
            'Sadness':2,
            'Anger':3,
            'Surprise':4,
            'Fear': 5,
            'Disgust': 6
        }
        self.gender_map = {
            'Male':0,
            'Female':1
        }
        # print(type(self.age_map))
    
    def __transform__(self, x):
        x = letterbox(x, (112, 112))
        x = np.transpose(x, (2,1,0))
        x = x[::-1]
        x = torch.from_numpy(x).float()
        x = x / 255.0
        if self.phase == 'train':
            x = self.train_transform(x)
        else:
            x = self.test_transform(x)

        return x
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.metadata.iloc[index]['file_name'])
        if not '.jpg'  in img_path:
            img_path += '.jpg'
        # print(img_path)
        img = cv2.imread(img_path)
        
        img = self.__transform__(img)
        # age = torch.tensor([1]*(self.age_map[self.metadata.iloc[index]['age']]) + [0]*(len(self.age_map.values()) - self.age_map[self.metadata.iloc[index]['age']])).float()
        age = self.age_map[self.metadata.iloc[index]['age']]
        gender = self.gender_map[self.metadata.iloc[index]['gender']]
        masked = self.masked_map[self.metadata.iloc[index]['masked']]
        emotion = self.emotion_map[self.metadata.iloc[index]['emotion']]
        race = self.race_map[self.metadata.iloc[index]['race']]
        skin = self.skintone_map[self.metadata.iloc[index]['skintone']]
        
        return img, age, gender, masked, emotion, race, skin
    
import random 
import torchvision.transforms.functional as TF
class RandomGammaCorrection(object):
    '''
    Apply Gamma Correction to the images
    '''

    def __init__(self, gamma = None):
        self.gamma = gamma
        
        
    def __call__(self,image):
        
        if self.gamma == None:
            # more chances of selecting 0 (original image)
            gammas = [0,0,0,0.5,1,1.5]
            self.gamma = random.choice(gammas)
        
        # print(self.gamma)
        if self.gamma == 0:
            return image
        
        else:
            return TF.adjust_gamma(image, self.gamma, gain=1)
        
def cutout(mask_size= 30, p= 0.5, cutout_inside= True, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = image.clone()

        if np.random.random() > p:
            return image

        h, w = image.shape[-2:]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[...,ymin:ymax, xmin:xmax] = torch.tensor(mask_color).view(1,3,1,1)
        return image

    return _cutout