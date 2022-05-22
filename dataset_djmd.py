import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils.jpeg import JPEG
import random
from preprocess_jpeg import load_file

class DCTDataset(Dataset):
    def __init__(self, filename="dataset", q=50) -> None:
        super().__init__()
        self.data = load_file("data", filename)
        self.q = q
        self.jpeg = JPEG(q, False)
    
    def __getitem__(self, index):
        x = y = self.data["x"][index]
        q_arr = [self.jpeg.quanti(item, idx>3) for idx, item in enumerate(x)]
        x = np.array(q_arr)
        return x, np.array(y)
        
    def __len__(self):
        return len(self.data["x"])
