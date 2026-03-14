import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# read image
path = r"C:\ntu_csie\1141\cv\lena.bmp"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
h, w = img.shape

# create output dir
output_path = r"C:\ntu_csie\1141\cv\output\hw4"
os.makedirs(output_path, exist_ok=True) 

# Binarize Lena with the threshold 128
threshold = 128
bi = np.zeros_like(img)
for i in range(h):
    for j in range(w):
        bi[i][j] = 255 if img[i][j] >= threshold else 0


# 3-5-5-5-3 kernel
kernel = np.array([
    [0,1,1,1,0],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [0,1,1,1,0]
], dtype=np.uint8)


# define dilation, erosion, openinng, and closing
def dilation(img, kernel):
    h, w = img.shape
    k_h, k_w = kernel.shape
    c_h, c_w = k_h // 2, k_w // 2
    out = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            if img[i][j] == 255:
                r_start, r_end = max(0, i-c_h), min(h, i+c_h+1)
                c_start, c_end = max(0, j-c_w), min(w, j+c_w+1)

                region = out[r_start:r_end, c_start:c_end]

                k_r_start = r_start - (i-c_h)   
                k_r_end   = k_r_start + (r_end - r_start)
                k_c_start = c_start - (j-c_w)
                k_c_end   = k_c_start + (c_end - c_start)

                region[kernel[k_r_start:k_r_end, k_c_start:k_c_end] == 1] = 255
    return out

def erosion(img, kernel):
    h, w = img.shape
    k_h, k_w = kernel.shape
    c_h, c_w = k_h // 2, k_w // 2
    out = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            if img[i][j] == 255:
                r_start, r_end = max(0, i-c_h), min(h, i+c_h+1)
                c_start, c_end = max(0, j-c_w), min(w, j+c_w+1)

                region = img[r_start:r_end, c_start:c_end]

                k_r_start = r_start - (i-c_h)   
                k_r_end   = k_r_start + (r_end - r_start)
                k_c_start = c_start - (j-c_w)
                k_c_end   = k_c_start + (c_end - c_start)

                if np.all(region[kernel[k_r_start:k_r_end, k_c_start:k_c_end] == 1] == 255):
                    out[i, j] = 255
    return out

def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)

def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)

def hit_and_miss(img, kernel):
    h, w = img.shape
    k_h, k_w = kernel.shape
    c_h, c_w = k_h // 2, k_w // 2
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            r_start, r_end = max(0, i-c_h), min(h, i+c_h+1)
            c_start, c_end = max(0, j-c_w), min(w, j+c_w+1)

            region = img[r_start:r_end, c_start:c_end]

            k_r_start = r_start - (i-c_h)   
            k_r_end   = k_r_start + (r_end - r_start)
            k_c_start = c_start - (j-c_w)
            k_c_end   = k_c_start + (c_end - c_start)

            match = True
            k_region = kernel[k_r_start:k_r_end, k_c_start:k_c_end]
            # j
            if np.any(region[kernel[k_r_start:k_r_end, k_c_start:k_c_end] == 1] != 255):
                match = False
            # k
            if np.any(region[kernel[k_r_start:k_r_end, k_c_start:k_c_end] == -1] != 0):
                match = False
            if match:
                out[i, j] = 255
    return out


# (a) Dilation
a = dilation(bi, kernel)
cv2.imwrite(os.path.join(output_path, "a.bmp"), a)

# (b) Erosion
b = erosion(bi, kernel)
cv2.imwrite(os.path.join(output_path, "b.bmp"), b)

# (c) Opening
c = opening(bi, kernel)
cv2.imwrite(os.path.join(output_path, "c.bmp"), c)

# (d) Closing
d = closing(bi, kernel)
cv2.imwrite(os.path.join(output_path, "d.bmp"), d)

# (e) Hit-and-miss transform
kernel = np.array([
    [0, -1, -1],
    [1, 1, -1],
    [0, 1, 0]
], dtype=int)
e = hit_and_miss(bi, kernel)
cv2.imwrite(os.path.join(output_path, "e.bmp"), e)