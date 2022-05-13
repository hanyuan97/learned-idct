import torch
import numpy as np
from model import LIDCT, FCIDCT, FCCNNIDCT, DECNNIDCT, RESIDCT, RESJPEGDECODER
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
    parser.add_argument("-w", "--weight", type=str, default="")
    parser.add_argument("-sp", "--sample", type=str, default="444")
    parser.add_argument("-l", "--lr", action="store_true")
    parser.add_argument("-g", "--gray", action="store_true")
    args = parser.parse_args()
    
    model_type = "res_dec"
    qf = args.qf
    
    jpeg = JPEG(qf, not args.gray and args.sample == "444")
    size = 8
    q_str = ""
    if qf > 0:
        q_str = str(qf)
    elif qf == -1:
        q_str = "random"
    weight = args.weight
    save_path = f"./jpeg_result/color_q{q_str}_s{args.sample}"
    crop_size = 8 if args.sample == "444" else 16
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
    elif model_type == "res_dec":
        model = RESJPEGDECODER(sample=args.sample)

    model.to(device)
    model.load_state_dict(torch.load(f"./weights/{weight}.pth"))
    model.eval()
    # test_set = load_file("data", filename)
    # print(len(dctDataset))

    data_path = "../dataset/DIV2K_valid_HR"
    # test_images = ["0848_4x_lr.png", "0850_4x_lr.png", "0869_4x_lr.png", "0900_4x_lr.png"]
    if args.lr:
        test_images = [i for i in os.listdir(data_path) if i.endswith("lr.png")]
    else:
        test_images = os.listdir(data_path)
    a_jpg_mse = 0
    a_jpg_psnr = 0
    a_res_mse = 0
    a_res_psnr = 0
    test_images = ["0805_4x_lr.png"]
    for name in test_images:
        if args.gray:
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
        else:
            img = np.float32(cv2.cvtColor(cv2.imread(f"{data_path}/{name}"), cv2.COLOR_BGR2YCR_CB))
            w = img.shape[1]
            h = img.shape[0]
            batch_size = 100
            img = img[:h//crop_size*crop_size, :w//crop_size*crop_size]
            jpeg_recon = np.zeros(img.shape)
            res_recon = np.zeros(img.shape)
            quan_recon = []
            if args.sample == "444":
                for y in range(0, img.shape[0] - crop_size + 1, crop_size):
                    for x in range(0, img.shape[1] - crop_size + 1, crop_size):
                        mcu = img[y:y+crop_size,x:x+crop_size].copy()
                        dct = jpeg.dct(mcu)
                        quan = jpeg.quanti(dct)
                        chw = quan.transpose(2, 0, 1)
                        yy = chw[0].copy()
                        cr = chw[1].copy()
                        cb = chw[2].copy()
                        quan_recon.append(np.concatenate((yy.reshape(64, 1, 1), cr.reshape(64, 1, 1), cb.reshape(64, 1, 1))))
                        decoded_mcu = jpeg.decode_mcu([yy, cr, cb])
                        jpeg_recon[y:y+crop_size,x:x+crop_size] = decoded_mcu
            else:
                print("420")
                for y in range(0, img.shape[0] - crop_size + 1, crop_size):
                    for x in range(0, img.shape[1] - crop_size + 1, crop_size):
                        mcu_arr = jpeg.split_16_ycrcb(img[y:y+crop_size,x:x+crop_size].copy())
                        dct_arr = [jpeg.dct(i) for i in mcu_arr]
                        qua_arr = [jpeg.quanti(item, idx>3) for idx, item in enumerate(dct_arr)]
                        iqua_arr = [jpeg.iquanti(item, idx>3) for idx, item in enumerate(qua_arr)]
                        idct_arr = [jpeg.idct(i) for i in iqua_arr]
                        quan_recon.append(np.concatenate([i.reshape(64, 1, 1) for i in qua_arr]))
                        jpeg_recon[y:y+crop_size,x:x+crop_size, 1] = cv2.resize(idct_arr[4], (16, 16))
                        jpeg_recon[y:y+crop_size,x:x+crop_size, 2] = cv2.resize(idct_arr[5], (16, 16))
                        jpeg_recon[y:y+8,x:x+8, 0] = idct_arr[0]
                        jpeg_recon[y:y+8,x+8:x+16, 0] = idct_arr[1]
                        jpeg_recon[y+8:y+16,x:x+8, 0] = idct_arr[2]
                        jpeg_recon[y+8:y+16,x+8:x+16, 0] = idct_arr[3]
                
            jpeg_recon[np.where(jpeg_recon > 255)] = 255
            jpeg_recon[np.where(jpeg_recon < 0)] = 0
            ipt = torch.from_numpy(np.array(quan_recon)[:batch_size]).to(device, dtype=torch.float)
            opt = model(ipt)
            c=0
            start = 0
            for y in range(img.shape[0]//crop_size):
                for x in range(img.shape[1]//crop_size):
                    if (c % batch_size) == 0:
                        start = c//batch_size * batch_size
                        ipt = torch.from_numpy(np.array(quan_recon)[start:start + batch_size]).to(device, dtype=torch.float)
                        opt = model(ipt)
                    test = opt[y*img.shape[1]//crop_size+x - start].cpu().detach().numpy()*255
                    res_recon[y*crop_size:y*crop_size+crop_size,x*crop_size:x*crop_size+crop_size] = test.transpose(1,2,0)
                    c+=1
              
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_YCR_CB2BGR)
        jpeg_recon = cv2.cvtColor(jpeg_recon.astype('uint8'), cv2.COLOR_YCR_CB2BGR)
        res_recon = cv2.cvtColor(res_recon.astype('uint8'), cv2.COLOR_YCR_CB2BGR)
         
        jpg_mse = np.mean((jpeg_recon - img)**2)
        res_mse = np.mean((res_recon - img)**2)
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
    log_file.write(f"----------------------------------\n")
    log_file.write(f"{a_jpg_mse},")
    log_file.write(f"{a_res_mse},")
    log_file.write(f"{a_jpg_psnr},")
    log_file.write(f"{a_res_psnr}\n")
    log_file.close()