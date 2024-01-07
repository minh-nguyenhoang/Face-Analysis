from torch.utils.data import Dataset, DataLoader
import albumentations
import torch
from .data_process.letterbox import letterbox
import pandas as pd
import os
import cv2


class PixtaDataset(Dataset):
    def __init__(self, 
                 root = '.',
                 csv_file = '',
                 img_size = (224,224),
                 transform = None,) -> None:
        super().__init__()

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
            

        self.transform = transform

    def __len_(self):
        return len(self.metadata)
    
    def __getitem__(self, index):

        

        return 
