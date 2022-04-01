import torch
import numpy as np
from model import LIDCT, FCIDCT, FCCNNIDCT, DECNNIDCT, RESIDCT
import matplotlib.pyplot as plt
import cv2
from utils.jpeg import JPEG
from utils.metrics import psnr
import math
import os
import argparse

def psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--qf", type=int, default=1)
    args = parser.parse_args()
    
    model_type = "res"
    qf = args.qf
    modelname = f"jpeg_model_{qf}"
    jpeg = JPEG(qf)
    size = 8
    q_str = ""
    if qf > 0:
        q_str = str(qf)
    elif qf == -1:
        q_str = "random"
    save_path = f"./jpeg_result/{q_str}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_file = open(f"{save_path}/detail.log", "w")
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # dctDataset = DCTDataset(filename=filename)
    # test_loader = DataLoader(dataset=dctDataset,
    #                         batch_size=3,
    #                         shuffle=True,
    #                         num_workers=8,
    #                         pin_memory=True)
    if model_type == "fc":
        model = FCIDCT(size=size)
    elif model_type == "cnn":
        model = LIDCT()
    elif model_type == "fc_cnn":
        model = FCCNNIDCT(size=size)
    elif model_type == "decnn":
        model = DECNNIDCT(num_channels=64, size=size)
    elif model_type == "res":
        model = RESIDCT(num_channels=64, size=size)

    model.to(device)
    model.load_state_dict(torch.load(f"./weights/{modelname}.pth"))
    model.eval()
    # test_set = load_file("data", filename)
    # print(len(dctDataset))

    data_path = "../dataset/DIV2K_valid_HR"
    # test_images = ["0848_4x_lr.png", "0850_4x_lr.png", "0869_4x_lr.png", "0900_4x_lr.png"]
    test_images = [i for i in os.listdir(data_path) if i.endswith("lr.png")]
    a_jpg_mse = 0
    a_jpg_psnr = 0
    a_res_mse = 0
    a_res_psnr = 0
    for name in test_images:
        img = np.float32(cv2.imread(f"{data_path}/{name}", cv2.IMREAD_GRAYSCALE))
        img = img[:img.shape[0]//8*8, :img.shape[1]//8*8]
        w = img.shape[1]
        h = img.shape[0]
        jpeg_recon = np.zeros((h, w))
        res_recon = np.zeros((h, w))
        quan_recon = []
        for y in range(0, h - size + 1, size):
            for x in range(0, w - size + 1, size):
                mcu = img[y:y+size,x:x+size]
                quan = jpeg.encode_mcu(mcu)
                quan_recon.append(quan)
                decoded_mcu = jpeg.decode_mcu(quan)
                jpeg_recon[y:y+size,x:x+size] = decoded_mcu

        quan_recon = torch.from_numpy(np.array(quan_recon)).to(device, dtype=torch.float)
        ipt = quan_recon.reshape(-1, 64, 1, 1)
        opt = model(ipt)
        for y in range(h//8):
            for x in range(w//8):
                res_recon[y*8:y*8+size,x*8:x*8+size] = opt[y*w//8+x].cpu().detach().numpy()*255
        
        jpg_mse = np.sum((jpeg_recon - img)**2)
        res_mse = np.sum((res_recon - img)**2)
        jpg_psnr = psnr(img, jpeg_recon)
        res_psnr = psnr(img, res_recon)
        a_jpg_mse += jpg_mse
        a_jpg_psnr += jpg_psnr
        a_res_mse += res_mse
        a_res_psnr += res_psnr
        log_file.write(f"{name},")
        log_file.write(f"{w*h},")
        
        log_file.write(f"{jpg_mse},")
        log_file.write(f"{res_mse},")
        log_file.write(f"{jpg_psnr},")
        log_file.write(f"{res_psnr}\n")
        
        cv2.imwrite(f"{save_path}/{name}_jpeg_recon.png", jpeg_recon)
        cv2.imwrite(f"{save_path}/{name}_res_recon.png", res_recon)

    l = len(test_images)
    a_jpg_mse /= l
    a_jpg_psnr /= l
    a_res_mse /= l
    a_res_psnr /= l
    log_file.write(f"----------------------------------")
    log_file.write(f"{a_jpg_mse},")
    log_file.write(f"{a_res_mse},")
    log_file.write(f"{a_jpg_psnr},")
    log_file.write(f"{a_res_psnr}\n")
    log_file.close()