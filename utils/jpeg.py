import cv2
import numpy as np

class JPEG:
    def __init__(self, qf=1, color=False) -> None:
        self.Q_F = qf
        self.color = color
        self.Qy =    np.array([ [16, 11, 10, 16, 24 , 40 , 51 , 61 ],
                                [12, 12, 14, 19, 26 , 58 , 60 , 55 ],
                                [14, 13, 16, 24, 40 , 57 , 69 , 56 ],
                                [14, 17, 22, 29, 51 , 87 , 80 , 62 ],
                                [18, 22, 37, 56, 68 , 109, 103, 77 ],
                                [24, 35, 55, 64, 81 , 104, 113, 92 ],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99 ]])
        
        self.Qcrcb = np.array([ [17, 18, 24, 47, 99 , 99 , 99 , 99 ],
                                [18, 21, 26, 66, 99 , 99 , 99 , 99 ],
                                [24, 26, 56, 99, 99 , 99 , 99 , 99 ],
                                [47, 66, 99, 99, 99 , 99 , 99 , 99 ],
                                [99, 99, 99, 99, 99 , 99 , 99 , 99 ],
                                [99, 99, 99, 99, 99 , 99 , 99 , 99 ],
                                [99, 99, 99, 99, 99 , 99 , 99 , 99 ],
                                [99, 99, 99, 99, 99 , 99 , 99 , 99 ],])
    
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
        if self.color:
            # return np.stack((np.round(cv2.dct(mcu[:,:,0]-128)), np.round(cv2.dct(mcu[:,:,1]-128)), np.round(cv2.dct(mcu[:,:,2]-128))), axis=2)
            return np.dstack((np.round(cv2.dct(mcu[:,:,0]-128)), np.round(cv2.dct(mcu[:,:,1]-128)), np.round(cv2.dct(mcu[:,:,2]-128))))
        return np.round(cv2.dct(mcu-128))

    def idct(self, iquan):
        if self.color:
            return np.dstack((np.round(cv2.idct(iquan[:,:,0])) + 128, np.round(cv2.idct(iquan[:,:,1])) + 128, np.round(cv2.idct(iquan[:,:,2])) + 128))
        return np.round(cv2.idct(iquan)) + 128
    
    def quanti(self, dct):
        if self.color:
           return np.dstack((dct[:,:,0] / (self.Qy * self.Q_F), dct[:,:,1] / (self.Qcrcb * self.Q_F), dct[:,:,2] / (self.Qcrcb * self.Q_F)))
        return np.round(dct / (self.Qy * self.Q_F))
    
    def iquanti(self, quan):
        if self.color:
            yy = np.round(quan[0] * (self.Qy * self.Q_F))
            cr = np.round(cv2.resize(quan[1], (8, 8), interpolation=cv2.INTER_NEAREST) * (self.Qcrcb * self.Q_F))
            cb = np.round(cv2.resize(quan[2], (8, 8), interpolation=cv2.INTER_NEAREST) * (self.Qcrcb * self.Q_F))
            return np.dstack((yy, cr, cb))
        return np.round(quan * (self.Qy * self.Q_F))
    
    def setQF(self, qf):
        self.Q_F = qf
    
    def setColor(self, color):
        self.color = color