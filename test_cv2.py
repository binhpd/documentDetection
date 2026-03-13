import time
print("Starting cv2 import test...")
start = time.time()
import cv2
print(f"cv2 imported successfully in {time.time() - start:.2f} seconds!")
print(f"cv2 version: {cv2.__version__}")
