import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils.jpeg import JPEG
import random
from preprocess_jpeg import load_file

class DCTDataset(Dataset):
    def __init__(self, filename="dataset", ndims=1, q=50, sample="444", rgb_mode=False) -> None:
        super().__init__()
        self.data = load_file("data", filename)
        self.ndims = ndims
        self.q = q
        self.jpeg = JPEG(q, ndims==3 and sample=="444")
        self.sample = sample
        self.RGB_MODE = rgb_mode
    
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
        elif self.ndims == 3:
            y = self.data["y"][index]
            if self.RGB_MODE:
                y = y.transpose(1, 2, 0)*255
                y = cv2.cvtColor(y.astype('uint8'), cv2.COLOR_YCR_CB2BGR)
                y = y.transpose(2, 0, 1).astype('float')/255
            if self.sample == "420":
                if self.q == -1:
                    self.jpeg.setQF(random.randint(1, 10))
                
                q_arr = [self.jpeg.quanti(item, idx>3).reshape(64, 1, 1) for idx, item in enumerate(self.data["x"][index])]
                x = np.concatenate(q_arr)
                return x, y
            elif self.sample == "444":
                if self.q == 0:
                    x = self.data["x"][index].reshape(192, 1, 1)
                    return x, self.data["y"][index]
                elif self.q == -1:
                    self.jpeg.setQF(random.randint(1, 10))
                    
                x = self.jpeg.quanti(self.data["x"][index]).transpose(2, 0, 1) # CHW np
                x0 = x[0].reshape(64, 1, 1)
                x1 = x[1].reshape(64, 1, 1)
                x2 = x[2].reshape(64, 1, 1)
                x = np.concatenate((x0, x1, x2))
                    
                return x, y
        
        return self.data["x"][index], self.data["y"][index]
    
    def __len__(self):
        return len(self.data["x"])
