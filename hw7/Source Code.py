import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# read image
path = r"C:\ntu_csie\1141\cv\lena.bmp"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
h, w = img.shape

# create output dir
output_dir = r"C:\ntu_csie\1141\cv\output\hw7"
os.makedirs(output_dir, exist_ok=True)

# binarize
bi = np.zeros_like(img, dtype=int)
for i in range(h):
    for j in range(w):
        bi[i, j] = 1 if img[i, j] >= 128 else 0

# downsample
down_h, down_w = h // 8, w // 8
down = np.zeros((down_h, down_w), dtype=int)
for i in range(down_h):
    for j in range(down_w):
        down[i,j] = bi[i*8, j*8]

def h(b, c, d, e):
    if b == c and (d != b or e != b):
        return 1
    else:
        return 0
    
def valid(x, y, h, w):
    return 0 <= x < h and 0 <= y < w

def f(img, x, y):
    if img[x, y] == 0:
        return 0

    row, col = img.shape
    x0 = img[x, y]
    x1 = img[x, y+1] if valid(x, y+1, row, col) else 0
    x2 = img[x-1, y] if valid(x-1, y, row, col) else 0
    x3 = img[x, y-1] if valid(x, y-1, row, col) else 0
    x4 = img[x+1, y] if valid(x+1, y, row, col) else 0
    x5 = img[x+1, y+1] if valid(x+1, y+1, row, col) else 0
    x6 = img[x-1, y+1] if valid(x-1, y+1, row, col) else 0
    x7 = img[x-1, y-1] if valid(x-1, y-1, row, col) else 0
    x8 = img[x+1, y-1] if valid(x+1, y-1, row, col) else 0

    a1 = h(x0, x1, x6, x2)
    a2 = h(x0, x2, x7, x3)
    a3 = h(x0, x3, x8, x4)
    a4 = h(x0, x4, x5, x1)

    if (a1 + a2 + a3 + a4) == 1:
        return 0
    else:
        return 1

def mark_b_i(img):
    row, col = img.shape
    mark = np.full((row, col), ' ', dtype='<U1')
    for x in range(row):
        for y in range(col):
            if img[x, y] == 0:
                continue
            cnt = 0            
            for (r, c) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                if valid(r, c, row, col) and img[r, c] == 1:
                    cnt += 1
            mark[x, y] = 'i' if cnt == 4 else 'b'
    return mark

def pair_relationship(img):
    row, col = img.shape
    pq = np.full((row, col), ' ', dtype='<U1')
    for x in range(row):
        for y in range(col):
            if img[x, y] == ' ':
                continue
            if img[x, y] == 'b':
                has_i = False
                for (r, c) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                    if valid(r, c, row, col) and img[r, c] == 'i':
                        has_i = True
                        break
                pq[x, y] = 'p' if has_i else 'q'
            else:
                pq[x, y] = 'q'
    return pq

def connected_shrink(img, mark_pq):
    row, col = img.shape
    out = img.copy()
    removed = 0
    for x in range(row):
        for y in range(col):
            if img[x, y] == 0:
                continue
            if mark_pq[x, y] != 'p':
                continue
            if f(img, x, y) == 0:
                out[x, y] = 0
                removed += 1
    return out, removed

cur = down.copy()
while True:
    mark_ib = mark_b_i(cur)
    mark_pq = pair_relationship(mark_ib)
    new_img, removed = connected_shrink(cur, mark_pq)
    cur = new_img
    if removed == 0:
        break   

thin = (cur * 255).astype(np.uint8)
out_img_path = os.path.join(output_dir, "thin.png")
cv2.imwrite(out_img_path, thin)
print(f"Image saved: {out_img_path}")