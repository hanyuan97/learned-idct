import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import DCTDataset
from model import LIDCT, FCIDCT, FCCNNIDCT, DECNNIDCT, RESIDCT
from tqdm import tqdm 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def init(args):
    EPOCH, model_type, dataset, dct_size, train_batch, val_batch, train_ratio, works, ndims = args.epoch, args.model_type, args.dataset, args.dct_size, args.batch, args.val_batch, args.train_ratio, args.works, args.ndims
    dctDataset = DCTDataset(filename=dataset, ndims=ndims)
    train_num = int(len(dctDataset) * args.train_ratio)
    val_num = len(dctDataset) - train_num
    
    train_set, val_set = random_split(dctDataset, [train_num, val_num])
    training_loader = DataLoader(dataset=train_set,
                            batch_size=train_batch,
                            shuffle=True,
                            num_workers=1,
                            pin_memory=True)

    validation_loader = DataLoader(dataset=val_set,
                            batch_size=val_batch,
                            shuffle=True,
                            num_workers=1,
                            pin_memory=True)

    
    if model_type == "cnn":
        model = LIDCT()
    elif model_type == "fc":
        model = FCIDCT()
    elif model_type == "fc_cnn":
        model = FCCNNIDCT()
    elif model_type == "decnn":
        model = DECNNIDCT()
    elif model_type == "res":
        model = RESIDCT()
    
    
    if not os.path.exists("./loss_log"):
        os.mkdir("./loss_log")
    
    log_file = open(f"./loss_log/{model_type}_{args.dataset}.log", "w")
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    return model, loss_fn, optimizer, training_loader, validation_loader, log_file
    
    
def train(EPOCH, model, loss_fn, optimizer, training_loader, validation_loader, log_file):
    for epoch in range(1, EPOCH+1):
        print(f"Epoch: {epoch}/{EPOCH}")
        model.train()
        train_loss = train_one_epoch(epoch)
        model.eval()
        val_loss = val_one_epoch(epoch)
        train_loss = train_loss/len(training_loader.dataset)
        val_loss = val_loss/len(validation_loader.dataset)
        print(f"Training Loss: {train_loss:.6f} \tValidation Loss: {val_loss:.6f}")
        log_file.write(f"{train_loss:.6f},{val_loss:.6f}\n")
    log_file.close()
    return model
    
def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.
    for data, label in tqdm(training_loader):
        data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output*255, label*255)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def val_one_epoch(epoch_index):
    running_loss = 0.
    for data, label in tqdm(validation_loader):
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = loss_fn(output*255, label*255)
        running_loss += loss.item()
    return running_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=20)
    parser.add_argument("-m", "--model_type", type=str, default="cnn")
    parser.add_argument("-d", "--dataset", type=str, default="dataset")
    parser.add_argument("-b", "--batch", type=int, default=64)
    parser.add_argument("-vb", "--val_batch", type=int, default=32)
    parser.add_argument("-ds", "--dct_size", type=int, default=8)
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.8)
    parser.add_argument("-o", "--output_filename", type=str, default="model.pth")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-n", "--ndims", type=int, default=1)
    parser.add_argument("-w", "--works", type=int, default=1)
    args = parser.parse_args()
    
    model, loss_fn, optimizer, training_loader, validation_loader, log_file = init(args)
    
    train(args.epoch, model, loss_fn, optimizer, training_loader, validation_loader, log_file)
    
    if args.save:
        torch.save(model.state_dict(), args.output_filename)
    # for data, label in tqdm(validation_loader):
    #     data, label = data.to(device), label.to(device)
    #     output = model(data)
    #     print(output[1])
    #     break
    