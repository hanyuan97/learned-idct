import torch
import numpy as np
from model import LIDCT, FCIDCT, FCCNNIDCT, DECNNIDCT, RESIDCT, RESJPEGDECODER
import matplotlib.pyplot as plt
import cv2
from utils.jpeg import JPEG
from utils.metrics import cal_ssim, cal_ms_ssim, psnr
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--qf", type=int, default=1)
    parser.add_argument("-b", "--batch", type=int, default=100)
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
    save_path = f"./jpeg_result/color_q{q_str}_s{args.sample}_dec"
    crop_size = 8 if args.sample == "444" else 16
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_file = open(f"{save_path}/color_q{q_str}_s{args.sample}_dec_log.csv", "w")
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
    if args.lr:
        test_images = [i for i in os.listdir(data_path) if i.endswith("lr.png")]
    else:
        test_images = os.listdir(data_path)
    # test_images = ["0848_4x_lr.png", "0850_4x_lr.png", "0869_4x_lr.png", "0900_4x_lr.png"]
    a_res_yuv_mse = []
    a_res_bgr_mse = []
    a_res_yuv_psnr = []
    a_res_bgr_psnr = []
    a_res_yuv_ssim = []
    a_res_bgr_ssim = []
    a_res_yuv_ms_ssim = []
    a_res_bgr_ms_ssim = []
    log_file.write("filename,w*h,yuv_mse,bgr_mse,psnr_y,psnr_u,psnr_v,psnr_b,psnr_g,psnr_r,ssim_y,ssim_u,ssim_v,ssim_b,ssim_g,ssim_r,ms_ssim_y,ms_ssim_u,ms_ssim_,ms_ssim_b,ms_ssim_g,ms_ssim_r,psnr_yuv,psnr_bgr,ssim_yuv,ssim_bgr,ms_ssim_yuv,ms_ssim_bgr\n")
    for name in test_images:
        if args.gray:
            img = np.float32(cv2.imread(f"{data_path}/{name}", cv2.IMREAD_GRAYSCALE))
            img = img[:img.shape[0]//8*8, :img.shape[1]//8*8]
            w = img.shape[1]
            h = img.shape[0]
            res_recon = np.zeros((h, w))
            quan_recon = []
            for y in range(0, h - size + 1, size):
                for x in range(0, w - size + 1, size):
                    mcu = img[y:y+size,x:x+size]
                    quan = jpeg.encode_mcu(mcu)
                    quan_recon.append(quan)

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
            batch_size = args.batch
            img = img[:h//crop_size*crop_size, :w//crop_size*crop_size]
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
            else:
                for y in range(0, img.shape[0] - crop_size + 1, crop_size):
                    for x in range(0, img.shape[1] - crop_size + 1, crop_size):
                        mcu_arr = jpeg.split_16_ycrcb(img[y:y+crop_size,x:x+crop_size].copy())
                        dct_arr = [jpeg.dct(i) for i in mcu_arr]
                        qua_arr = [jpeg.quanti(item, idx>3) for idx, item in enumerate(dct_arr)]
                        quan_recon.append(np.concatenate([i.reshape(64, 1, 1) for i in qua_arr]))

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
              
        img_bgr = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_YCR_CB2BGR)
        res_bgr = cv2.cvtColor(res_recon.astype('uint8'), cv2.COLOR_YCR_CB2BGR)
        
        res_yuv_mse = np.mean((res_recon - img)**2)
        res_bgr_mse = np.mean((res_bgr - img_bgr)**2)
        
        res_yuv_psnr = psnr(img, res_recon)
        res_bgr_psnr = psnr(img_bgr, res_bgr)
        
        res_yuv_ssim = np.array([cal_ssim(img[:,:,i], res_recon[:,:,i]) for i in range(3)])
        res_bgr_ssim = np.array([cal_ssim(img_bgr[:,:,i], res_bgr[:,:,i]) for i in range(3)])
        
        res_yuv_ms_ssim = np.array([cal_ms_ssim(img[:,:,i], res_recon[:,:,i]) for i in range(3)])
        res_bgr_ms_ssim = np.array([cal_ms_ssim(img_bgr[:,:,i], res_bgr[:,:,i]) for i in range(3)])
        
        a_res_yuv_mse.append(res_yuv_mse)
        a_res_bgr_mse.append(res_bgr_mse)
        
        a_res_yuv_psnr.append(res_yuv_psnr)
        a_res_bgr_psnr.append(res_bgr_psnr)
        a_res_yuv_ssim.append(res_yuv_ssim)
        a_res_bgr_ssim.append(res_bgr_ssim)
        
        a_res_yuv_ms_ssim.append(res_yuv_ms_ssim)
        a_res_bgr_ms_ssim.append(res_bgr_ms_ssim)
        
        opt = f"{name},"
        opt += f"{w*h},"
        opt += f"{res_yuv_mse},"
        opt += f"{res_bgr_mse},"
        opt += f"{','.join(map(str, res_yuv_psnr.tolist()))},"
        opt += f"{','.join(map(str, res_bgr_psnr.tolist()))},"
        opt += f"{','.join(map(str, res_yuv_ssim.tolist()))},"
        opt += f"{','.join(map(str, res_bgr_ssim.tolist()))},"
        opt += f"{','.join(map(str, res_yuv_ms_ssim.tolist()))},"
        opt += f"{','.join(map(str, res_bgr_ms_ssim.tolist()))},"
        opt += f"{np.mean(res_yuv_psnr)},"
        opt += f"{np.mean(res_bgr_psnr)},"
        opt += f"{np.mean(res_yuv_ssim)},"
        opt += f"{np.mean(res_bgr_ssim)},"
        opt += f"{np.mean(res_yuv_ms_ssim)},"
        opt += f"{np.mean(res_bgr_ms_ssim)}"
        print(opt)
        log_file.write(opt+"\n")
        
        cv2.imwrite(f"{save_path}/{name}_res_{qf}.png", res_bgr)

    opt = f"----------------------------------\n"
    opt += "Average,x,"
    opt += f"{np.mean(a_res_yuv_mse)},"
    opt += f"{np.mean(a_res_bgr_mse)},"
    opt += f"{','.join(map(str, np.mean(a_res_yuv_psnr, axis=0).tolist()))},"
    opt += f"{','.join(map(str, np.mean(a_res_bgr_psnr, axis=0).tolist()))},"
    opt += f"{','.join(map(str, np.mean(a_res_yuv_ssim, axis=0).tolist()))},"
    opt += f"{','.join(map(str, np.mean(a_res_bgr_ssim, axis=0).tolist()))},"
    opt += f"{','.join(map(str, np.mean(a_res_yuv_ms_ssim, axis=0).tolist()))},"
    opt += f"{','.join(map(str, np.mean(a_res_bgr_ms_ssim, axis=0).tolist()))},"
    opt += f"{np.mean(a_res_yuv_psnr)},"
    opt += f"{np.mean(a_res_bgr_psnr)},"
    opt += f"{np.mean(a_res_yuv_ssim)},"
    opt += f"{np.mean(a_res_bgr_ssim)},"
    opt += f"{np.mean(a_res_yuv_ms_ssim)},"
    opt += f"{np.mean(a_res_bgr_ms_ssim)}"
    
    print(opt)
    log_file.write(opt)
    log_file.close()