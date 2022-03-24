import torch.nn as nn
import torch.nn.functional as F

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