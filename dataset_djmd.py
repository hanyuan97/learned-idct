import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils.jpeg import JPEG
import random
from preprocess_jpeg import load_file
from utils.djmd import *

class DCTDataset(Dataset):
    def __init__(self, filename="dataset", q=50) -> None:
        super().__init__()
        self.data = load_file("data", filename)
        del self.data["y"]
        self.q = q
        self.jpeg = JPEG(q, False)
        self.x = []
        self.y = []
        self.quan()
        
    def quan(self):
        for i in self.data['x']:
            self.x.append([self.jpeg.quanti(item, idx>3) for idx, item in enumerate(i)])
            self.y.append(i)
        SHIFT_X, SCALE_X = get_shift_scale_maxmin(self.x)
        with open(f"./weights/normalize_q{self.q}.data", "w") as file:
            file.write(f"{SHIFT_X},{SCALE_X}")
        self.x = shift_and_normalize(np.array(self.x), SHIFT_X, SCALE_X)
        self.y = shift_and_normalize(np.array(self.y), SHIFT_Y, SCALE_Y)
        
    def __getitem__(self, index):
        # x = y = self.data["x"][index]
        # q_arr = [self.jpeg.quanti(item, idx>3) for idx, item in enumerate(x)]
        # x = np.array(q_arr)
        return self.x[index], self.y[index]
        
    def __len__(self):
        return len(self.x)
