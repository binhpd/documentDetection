import cv2


class AdaptiveThreshold:
    """Nhị phân hóa thích nghi — xử lý ánh sáng không đồng nhất.

    Tính ngưỡng T(x,y) riêng cho từng pixel dựa trên vùng lân cận B×B:
        T(x,y) = μ_B(x,y) - C

    Hai chế độ:
      - MEAN:     μ_B = trung bình cộng     → nhanh, đơn giản
      - GAUSSIAN: μ_B = trung bình Gaussian  → mượt, chính xác hơn
    """

    def __init__(self, block_size=11, C=7, method="gaussian"):
        if block_size % 2 == 0:
            block_size += 1
        self.block_size = block_size
        self.C = C
        self.method = method

    def apply(self, gray_image):
        """Áp dụng adaptive thresholding lên ảnh grayscale.

        Args:
            gray_image: ảnh 1 kênh, uint8

        Returns:
            binary: ảnh nhị phân (0 hoặc 255)
        """
        adapt_method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if self.method == "gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )

        binary = cv2.adaptiveThreshold(
            gray_image,
            maxValue=255,
            adaptiveMethod=adapt_method,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=self.block_size,
            C=self.C,
        )

        return binary
