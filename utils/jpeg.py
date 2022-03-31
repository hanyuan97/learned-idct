import cv2
import numpy as np

class JPEG:
    def __init__(self, qf=1) -> None:
        self.Q_F = qf
        self.Qy = np.array([ [16, 11, 10, 16, 24 , 40 , 51 , 61 ],
                             [12, 12, 14, 19, 26 , 58 , 60 , 55 ],
                             [14, 13, 16, 24, 40 , 57 , 69 , 56 ],
                             [14, 17, 22, 29, 51 , 87 , 80 , 62 ],
                             [18, 22, 37, 56, 68 , 109, 103, 77 ],
                             [24, 35, 55, 64, 81 , 104, 113, 92 ],
                             [49, 64, 78, 87, 103, 121, 120, 101],
                             [72, 92, 95, 98, 112, 100, 103, 99 ]])
    
    def encode(self, img):
        result = np.zeros((img.shape[0], img.shape[1]))
        size = 8
        for y in range(0, img.shape[0] - size + 1, size):
            for x in range(0, img.shape[1] - size + 1, size):
                result[y:y+size, x:x+size] = self.encode_mcu(img[y:y+size, x:x+size])
        return result
    
    def decode(self, encoded):
        result = np.zeros((encoded.shape[0], encoded.shape[1]))
        size = 8
        for y in range(0, encoded.shape[0] - size + 1, size):
            for x in range(0, encoded.shape[1] - size + 1, size):
                result[y:y+size, x:x+size] = self.decode_mcu(encoded[y:y+size, x:x+size])
        return result
    
    def encode_mcu(self, mcu):
        dct = self.dct(mcu)
        quan = self.quanti(dct)
        return quan
    
    def decode_mcu(self, quan):
        iquan = self.iquanti(quan)
        idct = self.idct(iquan)
        return idct
    
    def dct(self, mcu):
        return np.round(cv2.dct(mcu-128))

    def idct(self, iquan):
        return np.round(cv2.idct(iquan)) + 128
    
    def quanti(self, dct):
        return np.round(dct / (self.Qy * self.Q_F))
    
    def iquanti(self, quan):
        return np.round(quan * (self.Qy * self.Q_F))
    
    def setQF(self, qf):
        self.Q_F = qf