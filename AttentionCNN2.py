import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
        nn.Linear(channels, channels // reduction),
        nn.ReLU(inplace=True),
        nn.Linear(channels // reduction, channels),
        nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BottleNeckSEBlock(nn.Module):

    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels * self.expansion, 1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.se = SEblock(out_channels * self.expansion, reduction)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False)
            nn.BatchNorm2d(out_channels * self.expansion)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += self.shortcut
        return F.relu(out)
class 
       
