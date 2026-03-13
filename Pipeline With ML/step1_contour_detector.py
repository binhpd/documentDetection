"""
Step 1c: Tìm 4 góc tài liệu — Contour + approxPolyDP
🟢 Image Processing

Chức năng:
- Tìm tất cả contour (đường bao khép kín) trên ảnh cạnh
- Sắp xếp theo diện tích giảm dần
- Xấp xỉ mỗi contour thành đa giác (Douglas-Peucker)
- Nếu đa giác có đúng 4 đỉnh → đó là viền tờ giấy

Input:  Ảnh nhị phân cạnh (H × W, 0/255)
Output: Mảng 4 toạ độ góc (4, 2) float32 — hoặc None
"""

import cv2
import numpy as np


class ContourCornerDetector:
    def __init__(self, approx_epsilon=0.02, top_n=5):
        """
        Args:
            approx_epsilon: Tỷ lệ epsilon cho approxPolyDP (% chu vi).
                            Nhỏ hơn → xấp xỉ chính xác hơn, nhiều đỉnh hơn.
                            Lớn hơn → xấp xỉ thô hơn, ít đỉnh hơn.
            top_n: Số contour lớn nhất được xét
        """
        self.approx_epsilon = approx_epsilon
        self.top_n = top_n

    def find_corners(self, edged_image):
        """Tìm 4 góc tài liệu bằng contour + approxPolyDP.
        
        Returns:
            corners: np.array shape (4, 2) float32 — hoặc None nếu không tìm được
        """
        contours, _ = cv2.findContours(
            edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sắp xếp theo diện tích giảm dần, chỉ xét top_n contour lớn nhất
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.top_n]

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.approx_epsilon * peri, True)

            # Nếu xấp xỉ ra đúng 4 đỉnh → đây là tờ giấy
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

        return None
