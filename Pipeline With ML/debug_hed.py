import cv2
import numpy as np
from step1_hed_detector import HEDDetector
import matplotlib.pyplot as plt

image_path = "image/fold/0004.jpg"
image = cv2.imread(image_path)
ratio = 500.0 / image.shape[0]
resized = cv2.resize(image, (int(image.shape[1] * ratio), 500))

detector = HEDDetector("models/deploy.prototxt", "models/hed_pretrained_bsds.caffemodel")
detector._load_model()
(H, W) = resized.shape[:2]
blob = cv2.dnn.blobFromImage(resized, scalefactor=1.0, size=(W, H), mean=(104.0, 119.0, 122.0), swapRB=False, crop=False)
detector.net.setInput(blob)
hed_output = detector.net.forward()[0, 0]
hed_output = (255 * hed_output).astype("uint8")

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title("HED Output (Raw)")
plt.imshow(hed_output, cmap='gray')

_, mask50 = cv2.threshold(hed_output, 50, 255, cv2.THRESH_BINARY)
plt.subplot(132)
plt.title("Threshold 50")
plt.imshow(mask50, cmap='gray')

_, mask150 = cv2.threshold(hed_output, 150, 255, cv2.THRESH_BINARY)
plt.subplot(133)
plt.title("Threshold 150")
plt.imshow(mask150, cmap='gray')
plt.savefig("debug_hed.png")
print("Saved debug_hed.png")
