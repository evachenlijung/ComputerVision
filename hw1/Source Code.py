import cv2
from PIL import Image
import numpy as np
import os
import math

# read image
path = r"C:\ntu_csie\1141\cv\lena.bmp"
img = cv2.imread(path, cv2.IMREAD_COLOR)
h, w, ch = img.shape

# create output dir
output_path = r"C:\ntu_csie\1141\cv\output\hw1"
os.makedirs(output_path, exist_ok=True) 



# Part1. Write a program to do the following requirement.

# (a) upside-down lena.bmp
a = np.zeros((h, w, ch), dtype=np.uint8)
for i in range(h):
    a[i] = img[h-1-i]
cv2.imwrite(os.path.join(output_path, "part1_a.bmp"), a)


# (b) right-side-left lena.bmp
b = np.zeros((h, w, ch), dtype=np.uint8)
for j in range(w):
    b[:, j] = img[:, w-1-j]
cv2.imwrite(os.path.join(output_path, "part1_b.bmp"), b)


# (c) diagonally flip lena.bmp
c = np.zeros((w, h, ch), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        c[j][i] = img[i][j]
cv2.imwrite(os.path.join(output_path, "part1_c.bmp"), c)



# Part2. Write a program or use software to do the following requirement.

# (d) rotate lena.bmp 45 degrees clockwise
d = Image.open(path)
rotated = d.rotate(-45, expand=False)
rotated.save(os.path.join(output_path, "part2_d.bmp"))


# (e) shrink lena.bmp in half
h2, w2 = h//2, w//2
e = np.zeros((h2, w2, ch), dtype=np.uint8)
for i in range(h2):
    for j in range(w2):
        patc = img[i*2:i*2+1, j*2:j*2+1, :]
        e[i][j] = np.mean(patc, axis=(0, 1))
cv2.imwrite(os.path.join(output_path, "part2_e.bmp"), e)


# (f) binarize lena.bmp at 128 to get a binary image
f = np.zeros((h, w), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        r, g, b = img[i][j].astype(int)
        f[i][j] = 0 if (r+g+b)//3 < 128 else 255
cv2.imwrite(os.path.join(output_path, "part2_f.bmp"), f)