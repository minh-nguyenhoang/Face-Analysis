import torch.nn.functional as F
import torch
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import argparse
from src.models.bionet import *
from train import *
import logging
import json

if __name__ == "__main__":
  # Experiment options
  parser = argparse.ArgumentParser()

  parser.add_argument('--data_path', type=str, default='dataset/workout_processed_data.json',
                        help='JSON file for data')
  parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
  parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
  parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu')
  parser.add_argument('--epochs', type=int, default=50, help='number of epoch(s) to train')
  parser.add_argument('--lr', type=float, default=3e-3, help='learning rate to train')

  args = parser.parse_args()
  device = args.device