"""
Pytorch implementation of GroupFace: https://arxiv.org/abs/2005.10497
"""


import torch
import torch.nn as nn
from .frelu import FReLU

class FC(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(FC, self).__init__()
        self.fc = nn.Linear(inplanes, outplanes)
        self.act = nn.ReLU()
        # self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        return self.act(x)

class FCL(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(FCL, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(inplanes, inplanes//2),
            nn.ReLU(),
            nn.Linear(inplanes//2, inplanes//4),
            nn.ReLU(),
            nn.Linear(inplanes//4, outplanes)
        )
        # self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        return 


class GDN(nn.Module):
    def __init__(self, inplanes, outplanes, intermediate_dim=256):
        super(GDN, self).__init__()
        self.fc1 = FCL(inplanes, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, outplanes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        intermediate = self.fc1(x)
        out = self.fc2(intermediate)
        # return intermediate, self.softmax(out)
        return intermediate, torch.softmax(out, dim=1)
    

class CDN(nn.Module):
    def __init__(self, inplanes, outplanes, intermediate_dim=256):
        super(CDN, self).__init__()
        self.fc1 = FCL(inplanes, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, outplanes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        intermediate = self.fc1(x)
        out = self.fc2(intermediate)
        # return intermediate, self.softmax(out)
        return intermediate, torch.sigmoid(out)    
