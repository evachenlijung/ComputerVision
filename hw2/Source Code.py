import cv2
import numpy as np
import os
from typing import Tuple
from collections import deque
import matplotlib.pyplot as plt

# read image
path = r"C:\ntu_csie\1141\cv\lena_hw2.bmp"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
h, w = img.shape

# create output dir
output_path = r"C:\ntu_csie\1141\cv\output\hw2"
os.makedirs(output_path, exist_ok=True) 

# (a) a binary image (threshold at 128)
thres = 128
a = np.zeros((h, w), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        if img[i][j] >= thres:
            a[i][j] = 255
cv2.imwrite(os.path.join(output_path, "a.bmp"), a)


# (b) a histogram
N = 256
cnt = np.zeros(N, dtype=np.int32)
for i in range(h):
    for j in range(w):
        cnt[img[i][j]] += 1

plt.figure(figsize=(8,4))
plt.bar(np.arange(N), cnt, width=1.0, color='gray')
plt.savefig(os.path.join(output_path, "b.png"))


# (c) connected components (regions with + at centroid, bounding box)
r_mv = [-1,0,0,1]
c_mv = [0,-1,1,0]
visited = np.full((h, w), False, dtype=bool)

def valid(x, y):
    return 0 <= x < h and 0 <= y < w

def bfs(x, y) -> Tuple[int, int, int, int, int, int]:
    row_min = col_min = 1e9
    row_max = col_max = -1
    row_value_sum = 0
    col_value_sum = 0
    pixels = 0

    q = deque([(x, y)])
    visited[x][y] = True
    while q:
        (x, y) = q.popleft()
        row_value_sum += x
        col_value_sum += y
        pixels += 1
        row_min = min(row_min, x)
        row_max = max(row_max, x)
        col_min = min(col_min, y)
        col_max = max(col_max, y)

        for i in range(len(r_mv)):
            row = x + r_mv[i]
            col = y + c_mv[i]
            if(valid(row, col) and a[row][col] and not visited[row][col]):
                q.append((row, col))
                visited[row][col] = True
    cenx = row_value_sum // pixels
    ceny = col_value_sum // pixels
    return pixels, row_min, row_max, col_min, col_max, cenx, ceny

output_img = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
for i in range(h):
    for j in range(w):
        if a[i][j] == 255 and not visited[i][j]:
            pixels, row_min, row_max, col_min, col_max, cenx, ceny = bfs(i, j)
            if pixels >= 500:
                # bounding box
                cv2.rectangle(output_img, (col_min, row_min), (col_max, row_max), (255, 0, 0), 2)

                # cross
                k = 7
                cv2.line(output_img, (ceny, cenx - k), (ceny, cenx + k), (0, 0, 255), 2)
                cv2.line(output_img, (ceny - k, cenx), (ceny + k, cenx), (0, 0, 255), 2)

cv2.imwrite(os.path.join(output_path, "c.bmp"), output_img)