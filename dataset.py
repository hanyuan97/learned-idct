from configparser import Interpolation
import os
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
from utils.jpeg import JPEG
import random
import cv2
from preprocess import load_file

class DCTDataset(Dataset):
    def __init__(self, filename="dataset", ndims=1, q=0) -> None:
        super().__init__()
        self.data = load_file("data", filename)
        self.ndims = ndims
        self.q = q
        self.jpeg = JPEG(q, ndims==3)
    
    def __getitem__(self, index):
        if self.ndims == 64:
            if self.q == -1:
                self.jpeg.setQF(random.randint(1, 10))
                x = self.jpeg.quanti(self.data["x"][index]).reshape(64, 1, 1)
            elif self.q > 0:
                x = self.jpeg.quanti(self.data["x"][index]).reshape(64, 1, 1)
            else:
                x = self.data["x"][index].reshape(64, 1, 1)
            return x, self.data["y"][index]
        
        if self.ndims == 3:
            if self.q == -1:
                self.jpeg.setQF(random.randint(1, 10))
                x = self.jpeg.quanti(self.data["x"][index]).transpose(2, 0, 1) # CHW np
                x0 = x[0].reshape(64, 1, 1)
                x1 = cv2.resize(x[1], (4, 4), interpolation=cv2.INTER_AREA).reshape(16, 1, 1)
                x2 = cv2.resize(x[2], (4, 4), interpolation=cv2.INTER_AREA).reshape(16, 1, 1)
                x = np.concatenate((x0, x1, x2))
            elif self.q > 0:
                x = self.jpeg.quanti(self.data["x"][index]).transpose(2, 0, 1) # CHW np
                x0 = x[0].reshape(64, 1, 1)
                x1 = cv2.resize(x[1], (4, 4), interpolation=cv2.INTER_AREA).reshape(16, 1, 1)
                x2 = cv2.resize(x[2], (4, 4), interpolation=cv2.INTER_AREA).reshape(16, 1, 1)
                x = np.concatenate((x0, x1, x2))
            else:
                x = self.data["x"][index].reshape(64, 1, 1)
            return x, self.data["y"][index]
        
        return self.data["x"][index], self.data["y"][index]
    
    def __len__(self):
        return len(self.data["x"])
