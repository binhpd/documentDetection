"""
Step 1a: Tiền xử lý ảnh (Image Preprocessing)
🟢 Image Processing

Chức năng:
- Resize ảnh về chiều cao chuẩn (500px) để tăng tốc xử lý
- Chuyển ảnh màu BGR → Grayscale
- Áp dụng Gaussian Blur để loại bỏ nhiễu

Input:  Ảnh màu BGR (H × W × 3, uint8)
Output: Ảnh xám đã blur (500 × W' × 1, uint8), ratio scale
"""

import cv2
import numpy as np


class Preprocessor:
    def __init__(self, target_height=500, blur_kernel=(5, 5)):
        """
        Args:
            target_height: Chiều cao chuẩn hoá (pixel)
            blur_kernel: Kích thước kernel Gaussian Blur
        """
        self.target_height = target_height
        self.blur_kernel = blur_kernel

    def resize(self, image):
        """Resize ảnh về chiều cao chuẩn, giữ tỷ lệ.
        
        Returns:
            resized: Ảnh đã resize
            ratio: Tỷ lệ thu nhỏ (original_height / target_height)
        """
        ratio = image.shape[0] / self.target_height
        new_width = int(image.shape[1] / ratio)
        resized = cv2.resize(image, (new_width, self.target_height))
        return resized, ratio

    def to_grayscale(self, image):
        """Chuyển ảnh BGR → Grayscale (1 kênh)."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def blur(self, gray_image):
        """Áp dụng Gaussian Blur để giảm nhiễu."""
        return cv2.GaussianBlur(gray_image, self.blur_kernel, 0)

    def process(self, image, save_prefix=None):
        """Pipeline đầy đủ: resize → grayscale → blur."""
        resized, ratio = self.resize(image)
        if save_prefix is not None: cv2.imwrite(f"{save_prefix}_step1_0a_resized.jpg", resized)
        
        gray = self.to_grayscale(resized)
        if save_prefix is not None: cv2.imwrite(f"{save_prefix}_step1_0b_gray.jpg", gray)
        
        blurred = self.blur(gray)
        return blurred, resized, ratio
