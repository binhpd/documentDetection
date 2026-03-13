import cv2
import numpy as np

# Giả lập 1 đường viền bị đứt
img = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(img, (10, 10), (90, 90), 255, 1)
# Create gap
img[10, 40:60] = 0

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Max area with gap:", max([cv2.contourArea(c) for c in contours]))

# HED thick edge
img = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(img, (10, 10), (90, 90), 255, 3) # Thicker
# Create gap
img[10:13, 40:60] = 0

# Test morphology close to seal gap
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    print("Max area after close:", max([cv2.contourArea(c) for c in contours]))
