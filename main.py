import torch.nn.functional as F
import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import argparse
from src.models.bionet import *
from train import trainer, accuracy
import logging
import json
from torch.utils.data import DataLoader

from src.utils.dataset import PixtaDataset
from src.utils.loss import multi_task_loss
import timm
from tqdm import tqdm

def checkpoint_filter_fn(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']

    out_dict = {}
    if 'visual.trunk.stem.0.weight' in state_dict:
        out_dict = {k.replace('visual.trunk.', ''): v for k, v in state_dict.items() if k.startswith('visual.trunk.')}
        if 'visual.head.proj.weight' in state_dict:
            out_dict['head.fc.weight'] = state_dict['visual.head.proj.weight']
            out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.proj.weight'].shape[0])
        elif 'visual.head.mlp.fc1.weight' in state_dict:
            out_dict['head.pre_logits.fc.weight'] = state_dict['visual.head.mlp.fc1.weight']
            out_dict['head.pre_logits.fc.bias'] = state_dict['visual.head.mlp.fc1.bias']
            out_dict['head.fc.weight'] = state_dict['visual.head.mlp.fc2.weight']
            out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.mlp.fc2.weight'].shape[0])
        return out_dict

    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        if 'grn' in k:
            k = k.replace('grn.beta', 'mlp.grn.bias')
            k = k.replace('grn.gamma', 'mlp.grn.weight')
            v = v.reshape(v.shape[-1])
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v

    return out_dict
def checkpoint_dino_filter(state_dict, model):
    pretrained_state_dict = state_dict
    model_keys = model.state_dict().keys()

    new_state_dict = {}
    for k, v in pretrained_state_dict.items():
      new_k = k.replace("backbone.model.", "")
      if new_k in model_keys:
        new_state_dict[new_k] = v
    # print(len(new_student_state_dict.keys()))

    return new_state_dict
  
if __name__ == "__main__":
  # Experiment options
  parser = argparse.ArgumentParser()

  parser.add_argument('--data_path', type=str, default='dataset/workout_processed_data.json',
                        help='JSON file for data')
  parser.add_argument('--bs', type=int, default=32, help='batch size to train')
  parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
  parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
  parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu')
  parser.add_argument('--epochs', type=int, default=50, help='number of epoch(s) to train')
  parser.add_argument('--lr', type=float, default=1e-3, help='learning rate to train')

  args = parser.parse_args()
  
  device = args.device
  
  epochs = args.epochs
  batch_size = args.bs
  train_dataset = PixtaDataset(root='/kaggle/input/cropped-face-ai-hackathon/cropped_data/cropped_data',
                       csv_file='/kaggle/input/cropped-face-ai-hackathon/train.csv', phase='train')
  test_dataset = PixtaDataset(root='/kaggle/input/cropped-face-ai-hackathon/cropped_data/cropped_data',
                       csv_file='/kaggle/input/cropped-face-ai-hackathon/test.csv', phase='test')
  
  train_dl = DataLoader(train_dataset, batch_size, num_workers=16)
  test_dl = DataLoader(test_dataset, batch_size, num_workers=4)
  
  backbone = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)
#   checkpoint = torch.load('convnext_b_checkpoint.pth')
  # print(checkpoint['teacher'].keys())
#   new_checkpoint = checkpoint_dino_filter(checkpoint['teacher'],backbone)
  # print(new_checkpoint.keys())
#   backbone.load_state_dict(new_checkpoint, strict=False)
#   print(backbone)
  backbone.head = nn.Identity()
  loss_func = multi_task_loss()

  model = BioNet(backbone, 1024, 512, fine_tune=True)
  model.to(device)
#   for x, _, _, _, _, _, _ in train_dl:
#     x = x.cuda()
#     print(model(x))
  
  trainer(epochs, model, loss_func, train_dl, test_dl, opt_fn=None, lr=args.lr, metric=accuracy, PATH='', device=device)
    
  
  
