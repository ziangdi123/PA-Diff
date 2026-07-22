from annotator.canny import CannyDetector
from annotator.uniformer import UniformerDetector
from annotator.util import resize_image, HWC3
import numpy as np
import cv2

img = np.load('D:\\data\\liver_raw\\A\\2XY-001_A_R_3.npy').astype(np.float32)/2000.0
source = np.load('D:\\data\\liver_raw\\N\\2XY-001_N_3.npy').astype(np.float32)/2000.0
img = np.clip(img, 0.0, 1.0) * 255.0
source = np.clip(source, 0.0, 1.0) * 255.0
img = img.astype(np.uint8)
source = source.astype(np.uint8)
img2 = HWC3(img)
source2 = HWC3(source)
img3 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
source3 = cv2.applyColorMap(source, cv2.COLORMAP_JET)
apply_canny = CannyDetector()
apply_uniformer = UniformerDetector()
canny = apply_canny(img, 20, 80)
seg = apply_uniformer(img2)
print(seg.dtype, seg.shape, seg.max())
print(canny.dtype, canny.shape, canny.max())
cv2.imshow('', np.concatenate((source3, img3), axis=1))
cv2.waitKey(0)