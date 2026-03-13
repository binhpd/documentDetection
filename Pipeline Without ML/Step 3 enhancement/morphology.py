import cv2
import numpy as np


class MorphologyCleaner:
    """Làm sạch ảnh nhị phân bằng phép toán hình thái học.

    Pipeline:
      1. Opening  = Erosion → Dilation : loại bỏ nhiễu chấm nhỏ (muối tiêu)
      2. Closing  = Dilation → Erosion : lấp đầy lỗ nhỏ trong nét chữ
    """

    def __init__(self, open_kernel_size=2, close_kernel_size=2):
        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (open_kernel_size, open_kernel_size)
        )
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_kernel_size, close_kernel_size)
        )

    def clean(self, binary_image):
        """Áp dụng Opening rồi Closing lên ảnh nhị phân.

        Args:
            binary_image: ảnh nhị phân (0 / 255), uint8

        Returns:
            cleaned: ảnh đã làm sạch
        """
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, self.open_kernel)

        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.close_kernel)

        return cleaned
