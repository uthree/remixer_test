import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from remixer.remixer import ReMixerImageClassificator

class ConvolutionalClassificator(nn.Module):
    """Some Information about ConvolutionalClassificator"""
    def __init__(self):
        super(ConvolutionalClassificator, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3) # 3x32x32 -> 16x30x30
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(16, 32, 3) # 16x30x30 -> 32x28x28
        self.gelu2 = nn.GELU()
        self.pool1 = nn.MaxPool2d(2, 2) # 32x28x28 -> 32x14x14
        self.conv3 = nn.Conv2d(32, 64, 3) # 32x14x14 -> 64x12x12
        self.gelu3 = nn.GELU()
        self.conv4 = nn.Conv2d(64, 128, 3) # 64x12x12 -> 128x10x10
        self.gelu4 = nn.GELU()
        self.pool2 = nn.MaxPool2d(2, 2) # 128x10x10 -> 128x5x5
        self.conv5 = nn.Conv2d(128, 256, 3) # 128x5x5 -> 256x3x3
        self.gelu5 = nn.GELU()
        self.conv6 = nn.Conv2d(256, 512, 3) # 256x3x3 -> 512x1x1
        self.fc1 = nn.Linear(512, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = self.gelu2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.gelu3(x)
        x = self.conv4(x)
        x = self.gelu4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.gelu5(x)
        x = self.conv6(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        return x
        
class RemixerClassificatorLarge(nn.Module):
    """Some Information about RemixerClassificator"""
    def __init__(self):
        super(RemixerClassificatorLarge, self).__init__()
        self.remixer = ReMixerImageClassificator(3, 32, 4, classes=10, dim=1024, num_layers=18)
    def forward(self, x):
        x = self.remixer(x)
        return x

class RemixerClassificatorBase(nn.Module):
    """Some Information about RemixerClassificator"""
    def __init__(self):
        super(RemixerClassificatorBase, self).__init__()
        self.remixer = ReMixerImageClassificator(3, 32, 4, classes=10, dim=512, num_layers=12)
    def forward(self, x):
        x = self.remixer(x)
        return x

class RemixerClassificatorSmall(nn.Module):
    """Some Information about RemixerClassificator"""
    def __init__(self):
        super(RemixerClassificatorSmall, self).__init__()
        self.remixer = ReMixerImageClassificator(3, 32, 4, classes=10, dim=256, num_layers=6)
    def forward(self, x):
        x = self.remixer(x)
        return x
    
