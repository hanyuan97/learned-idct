from fileinput import filename
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import DCTDataset
from model import LIDCT
from tqdm import tqdm
from preprocess import load_file
import matplotlib.pyplot as plt
import matplotlib.image as img

def val_one_epoch(epoch_index):
    running_loss = 0.
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = loss_fn(output*255, label*255)
        print(output[0][0].cpu().detach().numpy())
        running_loss += loss.item()
        break
    return running_loss

if __name__=="__main__":
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dctDataset = DCTDataset(filename="testset")
    test_loader = DataLoader(dataset=dctDataset,
                            batch_size=3,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True)
    model = LIDCT()
    model.to(device)
    model.load_state_dict(torch.load("model.pth"))
    loss_fn = nn.MSELoss()
    model.eval()
    test_set = load_file("data", "testset")
    # print(test_set["y"][0].shape)
    plt.imshow(test_set["y"][30][0]*255, cmap="gray", vmin=0, vmax=255)
    plt.show()
    # val_one_epoch(1)