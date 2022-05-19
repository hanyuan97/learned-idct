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
    parser.add_argument("-q", "--qf", type=int, default=50)
    parser.add_argument("-sp", "--sample", type=str, default="444")
    parser.add_argument("-l", "--lr", action="store_true")
    parser.add_argument("-g", "--gray", action="store_true")
    
    args = parser.parse_args()
    
    qf = args.qf
    
    jpeg = JPEG(qf, not args.gray and args.sample == "444")
    size = 8
    q_str = ""
    if qf > 0:
        q_str = str(qf)
    elif qf == -1:
        q_str = "random"
    save_path = f"./jpeg_result/color_q{q_str}_s{args.sample}_jpg"
    crop_size = 8 if args.sample == "444" else 16
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_file = open(f"{save_path}/color_q{q_str}_s{args.sample}_jpg_log.csv", "w")
  
    # test_set = load_file("data", filename)
    # print(len(dctDataset))

    data_path = "../dataset/DIV2K_valid_HR"
    if args.lr:
        test_images = [i for i in os.listdir(data_path) if i.endswith("lr.png")]
    else:
        test_images = os.listdir(data_path)
    # test_images = ["0848_4x_lr.png", "0850_4x_lr.png", "0869_4x_lr.png", "0900_4x_lr.png"]
    a_jpg_yuv_mse = []
    a_jpg_bgr_mse = []
    a_jpg_yuv_psnr = []
    a_jpg_bgr_psnr = []
    a_jpg_yuv_ssim = []
    a_jpg_bgr_ssim = []
    a_jpg_yuv_ms_ssim = []
    a_jpg_bgr_ms_ssim = []
    log_file.write("filename,w*h,yuv_mse,bgr_mse,psnr_y,psnr_u,psnr_v,psnr_b,psnr_g,psnr_r,ssim_y,ssim_u,ssim_v,ssim_b,ssim_g,ssim_r,ms_ssim_y,ms_ssim_u,ms_ssim_v,ms_ssim_b,ms_ssim_g,ms_ssim_r,psnr_yuv,psnr_bgr,ssim_yuv,ssim_bgr,ms_ssim_yuv,ms_ssim_bgr\n")
    for name in test_images:
        if args.gray:
            img = np.float32(cv2.imread(f"{data_path}/{name}", cv2.IMREAD_GRAYSCALE))
            img = img[:img.shape[0]//8*8, :img.shape[1]//8*8]
            w = img.shape[1]
            h = img.shape[0]
            jpeg_recon = np.zeros((h, w))
            for y in range(0, h - size + 1, size):
                for x in range(0, w - size + 1, size):
                    mcu = img[y:y+size,x:x+size]
                    quan = jpeg.encode_mcu(mcu)
                    decoded_mcu = jpeg.decode_mcu(quan)
                    jpeg_recon[y:y+size,x:x+size] = decoded_mcu
        
        else:
            img = np.float32(cv2.cvtColor(cv2.imread(f"{data_path}/{name}"), cv2.COLOR_BGR2YCR_CB))
            w = img.shape[1]
            h = img.shape[0]
            img = img[:h//crop_size*crop_size, :w//crop_size*crop_size]
            jpeg_recon = np.zeros(img.shape)
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
                        decoded_mcu = jpeg.decode_mcu([yy, cr, cb])
                        jpeg_recon[y:y+crop_size,x:x+crop_size] = decoded_mcu
            else:
                for y in range(0, img.shape[0] - crop_size + 1, crop_size):
                    for x in range(0, img.shape[1] - crop_size + 1, crop_size):
                        mcu_arr = jpeg.split_16_ycrcb(img[y:y+crop_size,x:x+crop_size].copy())
                        dct_arr = [jpeg.dct(i) for i in mcu_arr]
                        qua_arr = [jpeg.quanti(item, idx>3) for idx, item in enumerate(dct_arr)]
                        iqua_arr = [jpeg.iquanti(item, idx>3) for idx, item in enumerate(qua_arr)]
                        idct_arr = [jpeg.idct(i) for i in iqua_arr]
                        jpeg_recon[y:y+crop_size,x:x+crop_size, 1] = cv2.resize(idct_arr[4], (16, 16))
                        jpeg_recon[y:y+crop_size,x:x+crop_size, 2] = cv2.resize(idct_arr[5], (16, 16))
                        jpeg_recon[y:y+8,x:x+8, 0] = idct_arr[0]
                        jpeg_recon[y:y+8,x+8:x+16, 0] = idct_arr[1]
                        jpeg_recon[y+8:y+16,x:x+8, 0] = idct_arr[2]
                        jpeg_recon[y+8:y+16,x+8:x+16, 0] = idct_arr[3]
                
            jpeg_recon[np.where(jpeg_recon > 255)] = 255
            jpeg_recon[np.where(jpeg_recon < 0)] = 0

              
        img_bgr = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_YCR_CB2BGR)
        jpeg_bgr = cv2.cvtColor(jpeg_recon.astype('uint8'), cv2.COLOR_YCR_CB2BGR)

        jpg_yuv_mse = np.mean((jpeg_recon - img)**2)
        jpg_bgr_mse = np.mean((jpeg_bgr - img_bgr)**2)

        jpg_yuv_psnr = psnr(img, jpeg_recon)
        jpg_bgr_psnr = psnr(img_bgr, jpeg_bgr)

        
        jpg_yuv_ssim = np.array([cal_ssim(img[:,:,i], jpeg_recon[:,:,i]) for i in range(3)])
        jpg_bgr_ssim = np.array([cal_ssim(img_bgr[:,:,i], jpeg_bgr[:,:,i]) for i in range(3)])
        
        jpg_yuv_ms_ssim = np.array([cal_ms_ssim(img[:,:,i], jpeg_recon[:,:,i]) for i in range(3)])
        jpg_bgr_ms_ssim = np.array([cal_ms_ssim(img_bgr[:,:,i], jpeg_bgr[:,:,i]) for i in range(3)])
        
        a_jpg_yuv_mse.append(jpg_yuv_mse)
        a_jpg_bgr_mse.append(jpg_bgr_mse)

        a_jpg_yuv_psnr.append(jpg_yuv_psnr)
        a_jpg_bgr_psnr.append(jpg_bgr_psnr)
        a_jpg_yuv_ssim.append(jpg_yuv_ssim)
        a_jpg_bgr_ssim.append(jpg_bgr_ssim)
        
        a_jpg_yuv_ms_ssim.append(jpg_yuv_ms_ssim)
        a_jpg_bgr_ms_ssim.append(jpg_bgr_ms_ssim)
        
        opt = f"{name},"
        opt += f"{w*h},"
        opt += f"{jpg_yuv_mse},"
        opt += f"{jpg_bgr_mse},"
        opt += f"{','.join(map(str, jpg_yuv_psnr.tolist()))},"
        opt += f"{','.join(map(str, jpg_bgr_psnr.tolist()))},"
        opt += f"{','.join(map(str, jpg_yuv_ssim.tolist()))},"
        opt += f"{','.join(map(str, jpg_bgr_ssim.tolist()))},"
        opt += f"{','.join(map(str, jpg_yuv_ms_ssim.tolist()))},"
        opt += f"{','.join(map(str, jpg_bgr_ms_ssim.tolist()))},"
        opt += f"{np.mean(jpg_yuv_psnr)},"
        opt += f"{np.mean(jpg_bgr_psnr)},"
        opt += f"{np.mean(jpg_yuv_ssim)},"
        opt += f"{np.mean(jpg_bgr_ssim)},"
        opt += f"{np.mean(jpg_yuv_ms_ssim)},"
        opt += f"{np.mean(jpg_bgr_ms_ssim)}"
        print(opt)
        log_file.write(opt+"\n")
        
        cv2.imwrite(f"{save_path}/{name}_jpeg_{qf}.png", jpeg_bgr)

    opt = f"----------------------------------\n"
    opt += "Average,x,"
    opt += f"{np.mean(a_jpg_yuv_mse)},"
    opt += f"{np.mean(a_jpg_bgr_mse)},"
    opt += f"{','.join(map(str, np.mean(a_jpg_yuv_psnr, axis=0).tolist()))},"
    opt += f"{','.join(map(str, np.mean(a_jpg_bgr_psnr, axis=0).tolist()))},"
    opt += f"{','.join(map(str, np.mean(a_jpg_yuv_ssim, axis=0).tolist()))},"
    opt += f"{','.join(map(str, np.mean(a_jpg_bgr_ssim, axis=0).tolist()))},"
    opt += f"{','.join(map(str, np.mean(a_jpg_yuv_ms_ssim, axis=0).tolist()))},"
    opt += f"{','.join(map(str, np.mean(a_jpg_bgr_ms_ssim, axis=0).tolist()))},"
    opt += f"{np.mean(a_jpg_yuv_psnr)},"
    opt += f"{np.mean(a_jpg_bgr_psnr)},"
    opt += f"{np.mean(a_jpg_yuv_ssim)},"
    opt += f"{np.mean(a_jpg_bgr_ssim)},"
    opt += f"{np.mean(a_jpg_yuv_ms_ssim)},"
    opt += f"{np.mean(a_jpg_bgr_ms_ssim)}"
    
    print(opt)
    log_file.write(opt)
    log_file.close()