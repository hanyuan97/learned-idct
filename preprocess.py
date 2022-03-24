import argparse, logging
import cv2
import numpy as np
import os, glob
from scipy.fft import dct, idct
import pickle

def preprocess(image_paths, size=32, gray=False, max=100) -> None: 
    patches = []
    labels = []
    c=0
    for path in image_paths:
        c+=1
        img = np.float32(cv2.imread(path, cv2.IMREAD_GRAYSCALE)) / 255
        for i in range(0, img.shape[0] - size + 1, size):
            for j in range(0, img.shape[1] - size + 1, size):
                patches.append(cv2.dct(img[i:i+size, j:j+size]).reshape((1, size, size)))
                labels.append(img[i:i+size, j:j+size].reshape((1, size, size)))
                if (len(patches) >= max):
                    return patches, labels
    return patches, labels

def save_file(x, y, output_path, filename, size, max):
    obj = {'x': x, 'y': y}
    with open(output_path+f"/{filename}_{max}_{size}_cv_dct.pickle", "wb") as file:
        pickle.dump(obj, file)

def load_file(path, filename):
    with open(path+f"/{filename}.pickle", "rb") as file:
        return pickle.load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", type=str)
    parser.add_argument("-g", "--gray", action="store_true")
    parser.add_argument("-s", "--size", type=int, default=32)
    parser.add_argument("-m", "--max", type=int, default=100)
    parser.add_argument("-f", "--filename", type=str, default="dataset")
    parser.add_argument("-o", "--output_dir", type=str, default="data")
    args = parser.parse_args()
    image_paths = glob.glob(os.path.join(args.dataset_dir, "*.png"))
    assert len(image_paths) > 0, "No image found in directory"
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    x, y = preprocess(image_paths, args.size, args.gray, args.max)
    print(len(x))
    save_file(x, y, args.output_dir, args.filename, args.size, len(x))