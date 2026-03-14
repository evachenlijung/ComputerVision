import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# read image
path = r"C:\ntu_csie\1141\cv\lena.bmp"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# create output dir
output_path = r"C:\ntu_csie\1141\cv\output\hw5"
os.makedirs(output_path, exist_ok=True) 

# 3-5-5-5-3 kernel
kernel = np.array([
    [0,1,1,1,0],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [0,1,1,1,0]
], dtype=np.uint8)

# define gray-scale dilation, erosion, openinng, and closing
def dilation(img, kernel):
    h, w = img.shape
    k_h, k_w = kernel.shape
    c_h, c_w = k_h // 2, k_w // 2
    padded_image = np.pad(img, ((c_h, k_h - 1 - c_h), (c_w, k_w - 1 - c_w)), 
                          mode='constant', constant_values=0)
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            window = padded_image[i : i + k_h, j : j + k_w]
            masked_values = window[kernel]
            out[i, j] = np.max(masked_values)
    return out

def erosion(image, mask):
    h, w = img.shape
    k_h, k_w = kernel.shape
    c_h, c_w = k_h // 2, k_w // 2
    out = np.zeros_like(img)
    padded_image = np.pad(image, ((c_h, k_h - 1 - c_h), (c_w, k_w - 1 - c_w)), 
                          mode='constant', constant_values=255)    
    out = np.zeros_like(image, dtype=np.int32)
    for i in range(h):
        for j in range(w):
            window = padded_image[i : i + k_h, j : j + k_w]
            masked_values = window[mask]
            out[i, j] = np.min(masked_values)
    return out

def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)

def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)

# (a) Dilation
a = dilation(img, kernel)
cv2.imwrite(os.path.join(output_path, "a.bmp"), a)

# (b) Erosion
b = erosion(img, kernel)
cv2.imwrite(os.path.join(output_path, "b.bmp"), b)

# (c) Opening
c = opening(img, kernel)
cv2.imwrite(os.path.join(output_path, "c.bmp"), c)

# (d) Closing
d = closing(img, kernel)
cv2.imwrite(os.path.join(output_path, "d.bmp"), d)