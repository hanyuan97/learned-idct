import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.gdn import GDN

class LDCT(nn.Module):
    def __init__(self, num_channels=1) -> None:
        super(LDCT, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9//2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5//2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5//2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
class LIDCT(nn.Module):
    def __init__(self, num_channels=1) -> None:
        super(LIDCT, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=3//2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=3//2)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=3//2)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3_ = nn.Conv2d(128, 1, kernel_size=3, padding=3//2)
        
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=5, padding=5//2)
#         self.conv4_bn = nn.BatchNorm2d(128)
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=3//2)
#         self.conv5_bn = nn.BatchNorm2d(256)
#         self.conv6 = nn.Conv2d(256, num_channels, kernel_size=3, padding=3//2)
#         self.conv6_bn = nn.BatchNorm2d(1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.relu(self.conv1_bn(self.conv1(x)))
        x = self.relu(self.conv2_bn(self.conv2(x)))
        x = self.relu(self.conv3_bn(self.conv3(x)))
        #x = self.relu(self.conv4_bn(self.conv4(x)))
        #x = self.relu(self.conv5_bn(self.conv5(x)))
        x = self.relu(self.conv3_(x))
        return x

class FCIDCT(nn.Module):
    def __init__(self, num_channels=1, size=8) -> None:
        super(FCIDCT, self).__init__()
        self.size = size
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)
        #self.fc5 = nn.Linear(32, 16)
        #self.fc6 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, self.size, self.size, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        #x = self.relu(self.fc5(x))
        #x = self.relu(self.fc6(x))
        x = x.view(-1, 1, self.size, self.size)
        return x
    
class FCCNNIDCT(nn.Module):
    def __init__(self, num_channels=1, size=8) -> None:
        super(FCCNNIDCT, self).__init__()
        self.size = size
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        # self.fc4 = nn.Linear(128, 32)
        # self.fc5 = nn.Linear(32, 1)
        # self.fc6 = nn.Linear(128, 1)
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=3//2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=3//2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=3//2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, self.size, self.size, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        # x = self.relu(self.fc4(x))
        # x = self.relu(self.fc5(x))
        # x = self.relu(self.fc6(x))
        x = x.view(-1, 64, self.size, self.size)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv3(x))
        # x = self.relu(self.fc4(x))
        return x
    
    
    
class DECNNIDCT(nn.Module):
    def __init__(self, num_channels=64, size=8) -> None:
        super(DECNNIDCT, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(num_channels, 32, kernel_size=8, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=3//2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=3//2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv3(x))
        # x = self.relu(self.fc4(x))
        return x

class RESIDCT(nn.Module):
    def __init__(self, num_channels=64, size=8) -> None:
        super(RESIDCT, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(num_channels, 64, kernel_size=8, padding=0),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            GDN(64)
        )
        self.block8 = nn.Conv2d(64, 1, kernel_size=3, padding=3//2)
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7 + block1)
        return (torch.tanh(block8) + 1) / 2


class RESJPEGDECODER(nn.Module):
    def __init__(self, num_channels=64, size=8) -> None:
        super(RESJPEGDECODER, self).__init__()
        self.crcbblock = nn.Sequential(
            nn.ConvTranspose2d(32, 128, kernel_size=8, padding=0),
            nn.PReLU()
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(num_channels, 64, kernel_size=8, padding=0),
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
        self.block8 = nn.Conv2d(192, 3, kernel_size=3, padding=3//2)
    def forward(self, x):
        block1 = self.block1(x[:,:64])
        crcb = self.crcbblock(x[:,64:])
        cat1 = torch.cat((block1, crcb), 1)
        block2 = self.block2(cat1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7 + block1)
        return (torch.tanh(block8) + 1) / 2 

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