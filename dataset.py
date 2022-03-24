import os
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
# from torchvision import transforms

from preprocess import load_file

class DCTDataset(Dataset):
    def __init__(self, filename="dataset", ndims=1) -> None:
        super().__init__()
        self.data = load_file("data", filename)
        self.ndims = ndims
    # def _transform(self):
    #     transforms_list = [transforms.ToTensor()]
    #     return transforms.Compose(transforms_list)
    
    def __getitem__(self, index):
        if self.ndims == 64:
            x = self.data["x"][index].reshape(64, 1, 1)
            return x, self.data["y"][index]
        
        return self.data["x"][index], self.data["y"][index]
    
    def __len__(self):
        return len(self.data["x"])
