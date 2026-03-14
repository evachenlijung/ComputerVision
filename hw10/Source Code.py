import cv2
import numpy as np
import os

path = r"C:\ntu_csie\1141\cv\lena.bmp"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
h, w = img.shape

output_dir = r"C:\ntu_csie\1141\cv\output\hw10"
os.makedirs(output_dir, exist_ok=True)

# (a) Laplace Mask1 (0, 1, 0, 1, -4, 1, 0, 1, 0), Threshold = 15
a = np.full((h, w), 255, dtype=np.uint8)
kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0],
])
threshold = 15
for i in range(1,h-1):
    for j in range(1,w-1):
        region = img[i-1:i+2, j-1:j+2]
        if np.sum(kernel*region) >= threshold:
            a[i, j] = 0
savepath = os.path.join(output_dir, "a.bmp")
cv2.imwrite(savepath, a)



# (b) Laplace Mask2 (1, 1, 1, 1, -8, 1, 1, 1, 1), Threshold = 15
b = np.full((h, w), 255, dtype=np.uint8)
kernel = (1/3) * np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1],
])
threshold = 15
for i in range(1,h-1):
    for j in range(1,w-1):
        region = img[i-1:i+2, j-1:j+2]
        if np.sum(kernel*region) >= threshold:
            b[i, j] = 0
savepath = os.path.join(output_dir, "b.bmp")
cv2.imwrite(savepath, b)



# (c) Minimum variance Laplacian: 20
c = np.full((h, w), 255, dtype=np.uint8)
kernel = (1/3) * np.array([
    [2,-1,2],
    [-1,-4,-1],
    [2,-1,2],
])
threshold = 20
for i in range(1,h-1):
    for j in range(1,w-1):
        region = img[i-1:i+2, j-1:j+2]
        if np.sum(kernel*region) >= threshold:
            c[i, j] = 0
savepath = os.path.join(output_dir, "c.bmp")
cv2.imwrite(savepath, c)



# (d) Laplace of Gaussian: 3000
d = np.full((h, w), 255, dtype=np.uint8)
tmp = np.zeros((h, w), dtype=float)
kernel = np.array([
    [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
    [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
    [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
    [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
    [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
    [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
    [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
    [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
    [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
    [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
    [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]
])
threshold = 3000
for i in range(5,h-5):
    for j in range(5,w-5):
        region = img[i-5:i+6, j-5:j+6]
        tmp[i, j] = np.sum(kernel*region)
# 8鄰
neighbors = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]
for y in range(1, h-1):
    for x in range(1, w-1):
        val = tmp[y, x]
        is_edge = False
        if val > threshold:
            for dy, dx in neighbors:
                if tmp[y+dy, x+dx] < -threshold:
                    is_edge = True
                    break        
        if is_edge:
            d[y, x] = 0
savepath = os.path.join(output_dir, "d.bmp")
cv2.imwrite(savepath, d)



# (e) Difference of Gaussian: 1
# (inhibitory sigma=3, excitatory sigma=1, kernel size 11x11)
e = np.full((h, w), 255, dtype=np.uint8)
tmp = np.zeros((h, w), dtype=float)
kernel = np.array([
    [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
    [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
    [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
    [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
    [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
    [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
    [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
    [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
    [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
    [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
    [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]
])
threshold = 1
for i in range(5,h-5):
    for j in range(5,w-5):
        region = img[i-5:i+6, j-5:j+6]
        tmp[i, j] = np.sum(kernel*region)
for y in range(1, h-1):
    for x in range(1, w-1):
        val = tmp[y, x]
        is_edge = False
        if val > threshold:
            for dy, dx in neighbors:
                if tmp[y+dy, x+dx] < -threshold:
                    is_edge = True
                    break        
        if is_edge:
            e[y, x] = 0
savepath = os.path.join(output_dir, "e.bmp")
cv2.imwrite(savepath, e)