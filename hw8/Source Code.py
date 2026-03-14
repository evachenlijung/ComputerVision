import cv2
import numpy as np
import os

path = r"C:\ntu_csie\1141\cv\lena.bmp"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
h, w = img.shape

output_dir = r"C:\ntu_csie\1141\cv\output\hw8"
os.makedirs(output_dir, exist_ok=True)

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def compute_snr(original, test_img):
    original_f = original.astype(np.float64) / 255.0
    test_f = test_img.astype(np.float64) / 255.0
    mu = np.mean(original_f)
    signal_energy = np.mean((original_f - mu) ** 2)

    mu_n = np.mean(test_f - original_f)
    noise_energy = np.mean((test_f - original_f - mu_n) ** 2)
    return 20 * np.log10(np.sqrt(signal_energy) / np.sqrt(noise_energy))

def gaussian_noise(img, amplitude):
    noise = amplitude * np.random.normal(0, 1, img.shape)
    noisy = img.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def salt_pepper_noise(img, threshold):
    noisy = img.copy()
    rand = np.random.uniform(0, 1, img.shape)
    noisy[rand < threshold] = 0
    noisy[rand > 1 - threshold] = 255
    return noisy

def pad_image(img, pad_h, pad_w, mode='edge'):
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)

def box_filter(img, ksize):
    pad = ksize // 2
    padded = pad_image(img, pad, pad, mode='edge')
    out = np.zeros_like(img, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+ksize, j:j+ksize]
            out[i, j] = np.mean(region)
    return out

def median_filter(img, ksize):
    pad = ksize // 2
    padded = pad_image(img, pad, pad, mode='edge')
    out = np.zeros_like(img, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+ksize, j:j+ksize]
            out[i, j] = np.median(region)
    return out

def octagon_kernel():
    return np.array([
        [0,1,1,1,0],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [0,1,1,1,0]
    ], dtype=np.uint8)

def erosion(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = pad_image(img, ph, pw, mode='edge')
    out = np.zeros_like(img, dtype=np.uint8)
    coords = np.argwhere(kernel == 1)
    for i in range(h):
        for j in range(w):
            vals = []
            for dy, dx in coords:
                yy = i + dy
                xx = j + dx
                vals.append(padded[yy, xx])
            out[i, j] = np.min(vals)
    return out

def dilation(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = pad_image(img, ph, pw, mode='edge')
    out = np.zeros_like(img, dtype=np.uint8)
    coords = np.argwhere(kernel == 1)
    for i in range(h):
        for j in range(w):
            vals = []
            for dy, dx in coords:
                yy = i + dy
                xx = j + dx
                vals.append(padded[yy, xx])
            out[i, j] = np.max(vals)
    return out

def opening_then_closing(img):
    k = octagon_kernel()
    eroded = erosion(img, k)
    opened = dilation(eroded, k)
    dilated = dilation(opened, k)
    result = erosion(dilated, k)
    return result

def closing_then_opening(img):
    k = octagon_kernel()
    dilated = dilation(img, k)
    closed = erosion(dilated, k)
    eroded = erosion(closed, k)
    result = dilation(eroded, k)
    return result

# 產生 4 張 noisy images
noise_configs = [
    ("gaussian_10", "Gaussian (amp=10)", lambda: gaussian_noise(img, 10)),
    ("gaussian_30", "Gaussian (amp=30)", lambda: gaussian_noise(img, 30)),
    ("saltpepper_005", "Salt & Pepper (p=0.05)", lambda: salt_pepper_noise(img, 0.05)),
    ("saltpepper_01", "Salt & Pepper (p=0.1)", lambda: salt_pepper_noise(img, 0.1)),
]

snr_lines = ["ImageName\tSNR_dB\n"]
all_results = {}

for key, label, func in noise_configs:
    noisy = func()
    snr = compute_snr(img, noisy)
    fname = f"{key}.bmp"
    cv2.imwrite(os.path.join(output_dir, fname), noisy)
    snr_lines.append(f"{key}\t{snr:.6f}\n")
    all_results[key] = {"label": label, "noisy": noisy, "snr_noisy": snr, "processed": {}}

    processed_methods = [
        ("box3", "Box 3x3", lambda im: box_filter(im, 3)),
        ("box5", "Box 5x5", lambda im: box_filter(im, 5)),
        ("median3", "Median 3x3", lambda im: median_filter(im, 3)),
        ("median5", "Median 5x5", lambda im: median_filter(im, 5)),
        ("open_close", "Opening then Closing", opening_then_closing),
        ("close_open", "Closing then Opening", closing_then_opening),
    ]

    for mkey, mlabel, mfunc in processed_methods:
        proc = mfunc(noisy)
        snr_p = compute_snr(img, proc)
        out_name = f"{key}_{mkey}.bmp"
        cv2.imwrite(os.path.join(output_dir, out_name), proc)
        snr_lines.append(f"{key}_{mkey}\t{snr_p:.6f}\n")
        all_results[key]["processed"][mkey] = {
            "label": mlabel,
            "img": proc,
            "snr": snr_p,
        }

snr_path = os.path.join(output_dir, "snr_values.txt")
with open(snr_path, "w") as f:
    f.writelines(snr_lines)