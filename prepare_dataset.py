import argparse
import cv2
import numpy as np
import os, glob
from numpy.random import randint
import pandas as pd

def prepare(dataset_dir, size=8, max=100) -> list:
    opt = []
    for filename in os.listdir(dataset_dir):
        image_path = f"{dataset_dir}/{filename}"
        img = cv2.imread(image_path)
        x = randint(0, int(img.shape[1]-size-1), max)
        y = randint(0, int(img.shape[0]-size-1), max)
        for i in range(max):
            opt.append(f"{filename},{x[i]},{x[i]+16},{y[i]},{y[i]+16}")
    return opt

def save_file(opt, output_path, filename, size, max):
    with open(output_path+f"/{filename}_{max}_{size}_color.csv", "w") as file:
        file.write('\n'.join(opt))

def load_file(path, filename):
    # "../dataset/DIV2K_train_HR"
    data = pd.read_csv(f"{path}/{filename}.csv").values
    data[0:, 0] = "../dataset/DIV2K_train_HR/" + data[0:, 0]
    return data
    # with open(path+f"/{filename}.pickle", "rb") as file:
    #     return pickle.load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", type=str)
    parser.add_argument("-s", "--size", type=int, default=8)
    parser.add_argument("-m", "--max", type=int, default=100)
    parser.add_argument("-f", "--filename", type=str, default="dataset")
    parser.add_argument("-o", "--output_dir", type=str, default="data")
    args = parser.parse_args()
    image_paths = glob.glob(os.path.join(args.dataset_dir, "*.png"))
    assert len(image_paths) > 0, "No image found in directory"
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    opt = prepare(args.dataset_dir, args.size, args.max)
    save_file(opt, args.output_dir, args.filename, args.size, len(opt))