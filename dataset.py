import os
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
from utils.jpeg import JPEG
import random

from preprocess import load_file

class DCTDataset(Dataset):
    def __init__(self, filename="dataset", ndims=1, q=0) -> None:
        super().__init__()
        self.data = load_file("data", filename)
        self.ndims = ndims
        self.q = q
        self.jpeg = JPEG(q)
    
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
        
        return self.data["x"][index], self.data["y"][index]
    
    def __len__(self):
        return len(self.data["x"])
