import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class CPAMEnc(nn.Module):
    """
    CPAM encoding module
    """
    def __init__(self, in_channels, norm_layer= nn.BatchNorm2d):
        super(CPAMEnc, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))

    def forward(self, x: Tensor):
        b, c, h, w = x.shape
        
        feat1 = self.conv1(self.pool1(x)).view(b,c,-1)
        feat2 = self.conv2(self.pool2(x)).view(b,c,-1)
        feat3 = self.conv3(self.pool3(x)).view(b,c,-1)
        feat4 = self.conv4(self.pool4(x)).view(b,c,-1)
        
        return torch.cat((feat1, feat2, feat3, feat4), 2).permute(0,2,1)


class CPAMDec(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key = nn.Linear(in_channels, in_channels//4) # key_conv2
        self.conv_value = nn.Linear(in_channels, in_channels) # value2

    def forward(self, x: Tensor, y: Tensor):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,M)
            returns :
                out : compact position attention feature
                attention map: (H*W)*M
        """
        m_batchsize,C,width ,height = x.shape
        m_batchsize,K,M = y.shape

        proj_query  = self.conv_query(x).view(m_batchsize,-1,width*height).permute(0,2,1)#BxNxd
        proj_key =  self.conv_key(y).view(m_batchsize,K,-1).permute(0,2,1)#BxdxK
        energy =  torch.bmm(proj_query,proj_key)#BxNxK
        attention = self.softmax(energy) #BxNxk

        proj_value = self.conv_value(y).permute(0,2,1) #BxCxK
        out = torch.bmm(proj_value,attention.permute(0,2,1))#BxCxN
        out = out.view(m_batchsize,C,width,height)
        out = self.scale*out + x
        return out

class CPAM(nn.Module):
    def __init__(self, in_channels, norm_layer= nn.BatchNorm2d, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pre_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
                                   norm_layer(in_channels // 4),
                                   nn.ReLU())
        self.encoder = CPAMEnc(in_channels // 4, norm_layer)
        self.decoder = CPAMDec(in_channels // 4)

    def forward(self, x: Tensor):
        out = self.pre_conv(x)

        attn = self.encoder(out)
        out = self.decoder(out, attn)

        return out


class CCAMDec(nn.Module):
    """
    CCAM decoding module
    """
    def __init__(self):
        super(CCAMDec,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor,y: Tensor):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,H,W)
            returns :
                out : compact channel attention feature
                attention map: K*C
        """
        m_batchsize,C,width ,height = x.shape
        x_reshape =x.view(m_batchsize,C,-1)

        B,K,W,H = y.shape
        y_reshape =y.view(B,K,-1)
        proj_query  = x_reshape #BXC1XN
        proj_key  = y_reshape.permute(0,2,1) #BX(N)XC
        energy =  torch.bmm(proj_query,proj_key) #BXC1XC
        energy_new = torch.max(energy,-1,keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = y.view(B,K,-1) #BCN
        
        out = torch.bmm(attention,proj_value) #BC1N
        out = out.view(m_batchsize,C,width ,height)

        out = x + self.scale*out
        return out
    
class CCAMEnc(nn.Module):
    """
    CCAM encoding module
    """
    def __init__(self, inter_channels, norm_layer = nn.BatchNorm2d) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(inter_channels, inter_channels//16, 1, bias=False),
                                   norm_layer(inter_channels//16),
                                   nn.ReLU())
        
    def forward(self, x):
        return self.layers(x)
        
class CCAM(nn.Module):
    def __init__(self, in_channels, norm_layer= nn.BatchNorm2d, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pre_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
                                   norm_layer(in_channels // 4),
                                   nn.ReLU())
        self.encoder = CCAMEnc(in_channels // 4, norm_layer)
        self.decoder = CCAMDec()

    def forward(self, x: Tensor):
        out = self.pre_conv(x)

        attn = self.encoder(out)
        out = self.decoder(out, attn)
        return out 
    
class DRAN(nn.Module):
    def __init__(self, in_channels, norm_layer = nn.BatchNorm2d) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.norm_layer = norm_layer
        self.ccam = CCAM(in_channels, norm_layer)
        self.cpam = CPAM(in_channels, norm_layer)
        self.post_ccam_Conv = nn.Sequential(nn.Conv2d(in_channels//4, in_channels // 4, 1, padding=0, bias=False),
                                   norm_layer(in_channels // 4),
                                   nn.ReLU())
        self.post_cpam_Conv = nn.Sequential(nn.Conv2d(in_channels//4, in_channels // 4, 1, padding=0, bias=False),
                                   norm_layer(in_channels // 4),
                                   nn.ReLU())
        self.fusion_conv = nn.Sequential(nn.Conv2d(in_channels//4*2, in_channels, 1, padding=0, bias=False),
                                   norm_layer(in_channels),
                                   nn.ReLU())
        
    def forward(self, x):
        ccam = self.ccam(x)
        cpam = self.cpam(x)

        ccam = self.post_ccam_Conv(ccam)
        cpam = self.post_cpam_Conv(cpam)

        feat = self.fusion_conv(torch.cat([cpam, ccam], dim = 1))

        return feat
    

