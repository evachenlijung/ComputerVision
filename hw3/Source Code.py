import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# read image
path = r"C:\ntu_csie\1141\cv\lena_hw3.bmp"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
h, w = img.shape
v = h*w

# create output dir
output_path = r"C:\ntu_csie\1141\cv\output\hw3"
os.makedirs(output_path, exist_ok=True) 


# (a) original image and its histogram
cv2.imwrite(os.path.join(output_path, "a_img.bmp"), img)

N = 256
cnt = np.zeros(N, dtype=np.int32)
for i in range(h):
    for j in range(w):
        cnt[img[i][j]] += 1
plt.figure(figsize=(8,4))
plt.bar(np.arange(N), cnt, width=1.0, color='gray')
plt.savefig(os.path.join(output_path, "a_hist.png"))


# (b) image with intensity divided by 3 and its histogram
b = np.zeros((h, w), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        b[i][j] = img[i][j] // 3
cv2.imwrite(os.path.join(output_path, "b_img.bmp"), b)

cnt = np.zeros(N, dtype=np.int32)
for i in range(h):
    for j in range(w):
        cnt[b[i][j]] += 1
plt.figure(figsize=(8,4))
plt.bar(np.arange(N), cnt, width=1.0, color='gray')
plt.savefig(os.path.join(output_path, "b_hist.png"))


# (c) image after applying histogram equalization to (b) and its histogram
pdf = np.zeros(N, dtype=np.float32) # Probability Distribution Function
cdf = np.zeros(N, dtype=np.float32) # Cumulative Distribution Function
for i in range(N):
    pdf[i] = cnt[i] / v
for i in range(1, N):
    cdf[i] = cdf[i-1] + pdf[i]
s = np.zeros(N, dtype=np.uint8) # transformation s = T(r)
for i in range(N):
    s[i] = round((N-1)*cdf[i])
c = np.zeros((h, w), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        c[i][j] = s[b[i][j]]
cv2.imwrite(os.path.join(output_path, "c_img.bmp"), c)

cnt = np.zeros(N, dtype=np.int32)
for i in range(h):
    for j in range(w):
        cnt[c[i][j]] += 1
plt.figure(figsize=(8,4))
plt.bar(np.arange(N), cnt, width=1.0, color='gray')
plt.savefig(os.path.join(output_path, "c_hist.png"))