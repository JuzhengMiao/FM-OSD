# networks

import math
from torch import nn
from torch.autograd import Function
import torch
import numpy as np

class Upnet_v3(nn.Module):
    def __init__(self, size, in_channels, out_channels = 128):
        super().__init__()
        self.size = size
        self.conv_out = nn.Conv2d(in_channels, out_channels, 3, padding = 1)

    def forward(self, x, num_patches):
        x = x.squeeze(1)
        b, n, c = x.shape
        x = x.reshape(b, num_patches[0], num_patches[1], c)
        x = x.permute(0, 3, 1, 2) 

        out = torch.nn.functional.interpolate(x, self.size, mode = 'bilinear')
        out = self.conv_out(out)
        return out

class Upnet_v3_coarsetofine2_tran_new(nn.Module): 
    def __init__(self, size, in_channels, out_channels = 128):
        super().__init__()
        self.size = size
        self.conv_out1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1)  # global
        self.conv_out2 = nn.Conv2d(out_channels, out_channels, 5, padding = 2)  # local
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
        )

    def forward(self, x, num_patches, size, islocal = False):
        x = x.squeeze(1)
        b, n, c = x.shape
        x = x.reshape(b, num_patches[0], num_patches[1], c)
        x = x.permute(0, 3, 1, 2)  
        if islocal:
            x = self.up(x)
            out = torch.nn.functional.interpolate(x, size, mode = 'bilinear')
            out_fine = self.conv_out2(out)
            return out_fine
        else:
            out = torch.nn.functional.interpolate(x, size, mode = 'bilinear')
            out_coarse = self.conv_out1(out)
            return out_coarse  
