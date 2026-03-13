import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
print("Starting main script import test...")
import cv2
cv2.setNumThreads(0)
print("cv2 imported.")
import torch
print("torch imported.")
from step1_ml_segmentor import MLSegmentor
print("MLSegmentor imported")
