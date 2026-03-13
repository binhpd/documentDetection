import cv2
import numpy as np


class ShadowRemoval:
    """Loại bỏ bóng đổ và cân bằng ánh sáng không đều.

    Thuật toán:
      1. Dilation với kernel lớn → ước lượng nền (background estimation)
      2. Trừ ảnh gốc cho nền: diff = dilated - original
      3. Chuẩn hóa và đảo ngược → ảnh sạch bóng, nền trắng đều
    """

    def __init__(self, kernel_size=7, morph_iterations=2):
        self.kernel_size = kernel_size
        self.morph_iterations = morph_iterations

    def remove(self, gray_image):
        """Loại bỏ bóng đổ khỏi ảnh grayscale.

        Args:
            gray_image: ảnh 1 kênh (grayscale), uint8

        Returns:
            shadow_free: ảnh đã loại bỏ bóng, nền trắng đều
            bg_estimate: ảnh nền ước lượng (để debug/hiển thị)
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )

        bg_estimate = cv2.dilate(gray_image, kernel, iterations=self.morph_iterations)
        bg_estimate = cv2.medianBlur(bg_estimate, 21)

        diff = cv2.absdiff(bg_estimate, gray_image)

        shadow_free = 255 - diff
        shadow_free = cv2.normalize(shadow_free, None, 0, 255, cv2.NORM_MINMAX)

        return shadow_free.astype(np.uint8), bg_estimate
