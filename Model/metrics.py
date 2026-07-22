import cv2
import os
import numpy as np
import pytorch_ssim
from skimage.metrics import structural_similarity as ssim
import torch

def psnr_gray(img1, img2):
    im1 = img1.astype(np.float32)
    im2 = img2.astype(np.float32)
    mse = np.mean((im1 - im2) ** 2)
    max_pixel_value = 255.0
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr_value


f1 = 'image_log_03_04\\train\\reconstruction_gs-241380_e-000072_b-000900.png'
f2 = 'image_log_03_04\\train\\samples_cfg_scale_9.00_gs-241380_e-000072_b-000900.png'
img1 = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)[2:514, 516:1028]
img2 = cv2.imread(f2, cv2.IMREAD_GRAYSCALE)[2:514, 516:1028]
cv2.imshow('', np.concatenate((img1, img2), axis=1))
cv2.waitKey(0)
print('PSNR: ', psnr_gray(img1, img2))
print('SSIM: ', ssim(img1, img2))
