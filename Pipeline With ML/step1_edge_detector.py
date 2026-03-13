"""
Step 1b: Phát hiện cạnh — Canny Edge Detection
🟢 Image Processing

Chức năng:
- Tự động tính ngưỡng Canny dựa trên giá trị median (thay vì cứng 75, 200)
- Phát hiện các pixel có gradient cường độ sáng đột biến (= cạnh)

Input:  Ảnh xám đã blur (H × W × 1, uint8, giá trị 0-255)
Output: Ảnh nhị phân cạnh (H × W × 1, uint8, giá trị 0 hoặc 255)
"""

import cv2
import numpy as np


class EdgeDetector:
    def __init__(self, sigma=0.33):
        """
        Args:
            sigma: Hệ số điều chỉnh ngưỡng tự động.
                   sigma nhỏ → ngưỡng hẹp → ít cạnh (chặt chẽ hơn)
                   sigma lớn → ngưỡng rộng → nhiều cạnh (nhạy hơn)
        """
        self.sigma = sigma

    def auto_canny(self, gray_image):
        """Canny Edge Detection với ngưỡng tự động dựa trên median.

        Công thức:
            median = median(pixel values)
            low  = max(0,   (1 - sigma) * median)
            high = min(255, (1 + sigma) * median)

        Ưu điểm so với ngưỡng cứng (75, 200):
            - Tự thích nghi với ảnh sáng/tối khác nhau
            - Ảnh tối (median thấp) → ngưỡng thấp → vẫn bắt được cạnh
            - Ảnh sáng (median cao) → ngưỡng cao → lọc bớt nhiễu
        """
        median = np.median(gray_image)
        low = int(max(0, (1.0 - self.sigma) * median))
        high = int(min(255, (1.0 + self.sigma) * median))
        edged = cv2.Canny(gray_image, low, high)
        return edged

    def detect(self, gray_image):
        """Phát hiện cạnh.
        
        Returns:
            edged: Ảnh nhị phân cạnh (0 hoặc 255)
        """
        return self.auto_canny(gray_image)
