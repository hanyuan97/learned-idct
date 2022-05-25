import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.gdn import GDN

class RESJPEGDECODER(nn.Module):
    def __init__(self, num_channels=64, sample="444") -> None:
        super(RESJPEGDECODER, self).__init__()
        
        self.ycrcbblock = nn.Sequential(
            nn.Conv2d(6, 192, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(192)
        self.block3 = ResidualBlock(192)
        self.block4 = ResidualBlock(192)
        self.block5 = ResidualBlock(192)
        self.block6 = ResidualBlock(192)
        self.block7 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            GDN(192)
        )
        self.block8 = nn.Conv2d(192, 6, kernel_size=3, padding=3//2)
        self.relu = nn.ReLU()
    def forward(self, x):
        cat1 = self.ycrcbblock(x)
        block2 = self.block2(cat1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7 + cat1)
        return self.relu(block8)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gdn1 = GDN(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gdn2 = GDN(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.gdn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.gdn2(residual)

        return x + residual