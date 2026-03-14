import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# read image
path = r"C:\ntu_csie\1141\cv\lena.bmp"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
height, width = img.shape

# create output dir
output_path = r"C:\ntu_csie\1141\cv\output\hw6"
os.makedirs(output_path, exist_ok=True)

# binarize
bi = np.zeros_like(img, dtype=int)
for i in range(height):
    for j in range(width):
        bi[i, j] = 1 if img[i, j] >= 128 else 0

# downsample
down_h, down_w = height // 8, width // 8
down = np.zeros((down_h, down_w), dtype=int)
for i in range(down_h):
    for j in range(down_w):
        down[i,j] = bi[i*8, j*8]

# count Yokoi connectivity number using 4-connected
def h(b, c, d, e):
    if b == c and (d != b or e != b):
        return 'q'
    elif b == c and (d == b and e == b):
        return 'r'
    else:
        return 's'

def yokoi(img, x, y):
    if img[x, y] == 0:
        return 0

    x0 = img[x, y]
    row, col = img.shape

    x1 = img[x, y+1] if y+1 < col else 0
    x2 = img[x-1, y] if x-1 >= 0 else 0
    x3 = img[x, y-1] if y-1 >= 0 else 0
    x4 = img[x+1, y] if x+1 < row else 0
    x5 = img[x+1, y+1] if x+1 < row and y+1 < col else 0
    x6 = img[x-1, y+1] if x-1 >= 0 and y+1 < col else 0
    x7 = img[x-1, y-1] if x-1 >= 0 and y-1 >= 0 else 0
    x8 = img[x+1, y-1] if x+1< row and y-1 >= 0 else 0

    a1 = h(x0, x1, x6, x2)
    a2 = h(x0, x2, x7, x3)
    a3 = h(x0, x3, x8, x4)
    a4 = h(x0, x4, x5, x1)

    if [a1, a2, a3, a4].count('r') == 4:
        return 5
    return [a1, a2, a3, a4].count('q')

yokoi_map = np.zeros((down_h, down_w), dtype=int)
for i in range(down_h):
    for j in range(down_w):
        yokoi_map[i, j] = yokoi(down, i, j)

# draw matrix
cell_size = 10  # pixel size per cell
font_scale = 0.35
thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX

img_h, img_w = down_h * cell_size, down_w * cell_size
out_img = np.full((img_h, img_w), 255, np.uint8)  # white background

for i in range(down_h):
    for j in range(down_w):
        v = yokoi_map[i, j]
        if v > 0:
            text = str(v)
            x = j * cell_size + 2
            y = (i + 1) * cell_size - 2
            cv2.putText(out_img, text, (x, y), font, font_scale, (0,), thickness, lineType=cv2.LINE_AA)

out_img_path = os.path.join(output_path, "yokoi_image.png")
cv2.imwrite(out_img_path, out_img)
print(f"Image saved: {out_img_path}")