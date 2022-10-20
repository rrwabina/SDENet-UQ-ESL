import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class CnvMod(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size = (5, 5)):
        super(CnvMod, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 
                        kernel_size = (1, 1), bias = True),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace = True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class EncMod(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(EncMod, self).__init__()
        self.block = nn.Sequential(
            CnvMod(input_channel, output_channel, kernel_size = (3, 3)),
            nn.MaxPool2d(1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DecMod(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DecMod, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel,
                                kernel_size = (2, 2), bias = True),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        return self.block(x)

class Map(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Map, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 
                        kernel_size = (4, 4), bias = True),
            nn.LogSigmoid()
        )
    def forward(self, x):
        return self.block(x)