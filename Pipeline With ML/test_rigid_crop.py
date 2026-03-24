import cv2
import numpy as np
import sys
import os

img = cv2.imread('../image/curved/0005.jpg')
from rembg import remove
subject_orig = remove(img)
mask = subject_orig[:, :, 3]
alpha_c = subject_orig[:, :, 3]
rgb_c = subject_orig[:, :, :3]
white_bg = np.ones_like(rgb_c) * 255
mask_f = alpha_c[:, :, np.newaxis] / 255.0
pure_doc = (rgb_c * mask_f + white_bg * (1 - mask_f)).astype(np.uint8)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
c = max(contours, key=cv2.contourArea)

rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)

from corner_sorter import sort_corners
src_pts = sort_corners(box)

width = int(np.linalg.norm(src_pts[0] - src_pts[1]))
height = int(np.linalg.norm(src_pts[0] - src_pts[3]))

if width > height:
    # Rotate 90 degrees if it's landscape by swapping target corners
    dst_pts = np.array([[width-1, 0],
                        [width-1, height-1],
                        [0, height-1],
                        [0, 0]], dtype="float32")
    width, height = height, width
else:
    dst_pts = np.array([[0, 0],
                        [width-1, 0],
                        [width-1, height-1],
                        [0, height-1]], dtype="float32")

M_crop = cv2.getPerspectiveTransform(src_pts, dst_pts)
cropped = cv2.warpPerspective(result['u2net_doc'], M_crop, (width, height), borderValue=(255,255,255))

cv2.imwrite('result/test_rigid_crop.jpg', cropped)
print("Saved test_crop.jpg", cropped.shape)
