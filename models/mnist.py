import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from remixer.remixer import ReMixerImageClassificator

class ConvolutionalClassificator(nn.Module):
    """Some Information about ConvolutionalClassificator"""
    def __init__(self):
        super(ConvolutionalClassificator, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3) # [1, 28, 28] -> [16, 26, 26]
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3) # [16, 26, 26] -> [32, 24, 24]
        self.gelu2 = nn.GELU()
        self.pool1 = nn.MaxPool2d(2, 2) # [32, 24, 24] -> [32, 12, 12]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3) # [32, 12, 12] -> [64, 10, 10]
        self.gelu3 = nn.GELU()
        self.pool2 = nn.MaxPool2d(2, 2) # [64, 10, 10] -> [64, 5, 5]
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3) # [64, 5, 5] -> [128, 3, 3]
        self.gelu4 = nn.GELU()
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3) # [128, 3, 3] -> [256, 1, 1]
        self.gelu5 = nn.GELU()
        self.lienar = nn.Linear(256, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = self.gelu2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.gelu3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.gelu4(x)
        x = self.conv5(x)
        x = self.gelu5(x)
        x = x.view(-1, 256)
        x = self.lienar(x)
        return x
        
class RemixerClassificatorLarge(nn.Module):
    """Some Information about RemixerClassificator"""
    def __init__(self):
        super(RemixerClassificatorLarge, self).__init__()
        self.remixer = ReMixerImageClassificator(1, 28, 7, classes=10, dim=1024, num_layers=18)
    def forward(self, x):
        x = self.remixer(x)
        return x

class RemixerClassificatorBase(nn.Module):
    """Some Information about RemixerClassificator"""
    def __init__(self):
        super(RemixerClassificatorBase, self).__init__()
        self.remixer = ReMixerImageClassificator(1, 28, 7, classes=10, dim=512, num_layers=12)
    def forward(self, x):
        x = self.remixer(x)
        return x

class RemixerClassificatorSmall(nn.Module):
    """Some Information about RemixerClassificator"""
    def __init__(self):
        super(RemixerClassificatorSmall, self).__init__()
        self.remixer = ReMixerImageClassificator(1, 28, 7, classes=10, dim=256, num_layers=6)
    def forward(self, x):
        x = self.remixer(x)
        return x
    
