import argparse, logging
import cv2
import numpy as np
import os, glob
import pickle
from utils.jpeg import JPEG
import random

jpeg = JPEG(qf=1)

def preprocess(image_paths, size=8, gray=False, max=100, dct=False) -> None: 
    jpeg.setColor(not gray and size==8)
    C = 1 if gray else 3
    patches = []
    labels = []
    for path in image_paths:
        if gray:
            img = np.float32(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        else:
            img = np.float32(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2YCR_CB))
        for i in range(max):
            x = random.randint(0, int(img.shape[1]-size-1))
            y = random.randint(0, int(img.shape[0]-size-1))
            mcu = img[y:y+size, x:x+size].copy()
            labels.append((mcu.copy()/255).transpose(2, 0, 1))
            if dct:
                if size == 16:
                    mcu_y = mcu[:, :, 0]
                    dct_y0 = jpeg.dct(mcu_y[:8, :8])
                    dct_y1 = jpeg.dct(mcu_y[:8, 8:])
                    dct_y2 = jpeg.dct(mcu_y[8:, :8])
                    dct_y3 = jpeg.dct(mcu_y[8:, 8:])
                    dct_cr = jpeg.dct(mcu[1::2, ::2, 1].copy())
                    dct_cb = jpeg.dct(mcu[0::2, ::2, 2].copy())
                    patches.append([dct_y0, dct_y1, dct_y2, dct_y3, dct_cr, dct_cb])
                else:
                    patches.append(jpeg.dct(mcu))
            else:
                patches.append(jpeg.encode_mcu(mcu))
    return patches, labels

def save_file(x, y, output_path, filename, size, max, gray):
    xData = np.array(x)
    yData = np.array(y)
    np.savez(f"{output_path}/{filename}_{max}_{size}{'' if gray else '_color'}.npz", x=xData, y=yData)
    
    # obj = {'x': x, 'y': y}
    # with open(output_path+f"/{filename}_{max}_{size}{'' if gray else '_color'}.pickle", "wb") as file:
    #     pickle.dump(obj, file)

def load_file(path, filename):
    return np.load(f"{path}/{filename}.npz")
    # with open(path+f"/{filename}.pickle", "rb") as file:
    #     return pickle.load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", type=str)
    parser.add_argument("-g", "--gray", action="store_true")
    parser.add_argument("--dct", action="store_true")
    parser.add_argument("-s", "--size", type=int, default=8)
    parser.add_argument("-m", "--max", type=int, default=100)
    parser.add_argument("-f", "--filename", type=str, default="dataset")
    parser.add_argument("-o", "--output_dir", type=str, default="data")
    args = parser.parse_args()
    image_paths = glob.glob(os.path.join(args.dataset_dir, "*.png"))
    assert len(image_paths) > 0, "No image found in directory"
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    x, y = preprocess(image_paths, args.size, args.gray, args.max, args.dct)
    save_file(x, y, args.output_dir, args.filename, args.size, len(x), args.gray)