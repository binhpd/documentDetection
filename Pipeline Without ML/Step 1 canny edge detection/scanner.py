import cv2
import numpy as np
from preprocessor import ImagePreprocessor
from edge_detector import EdgeDetector
from corner_detector import CornerDetector

class DocumentScanner:
    def __init__(self):
        # Khởi tạo các module con
        self.preprocessor = ImagePreprocessor()
        self.edge_detector = EdgeDetector()
        self.corner_detector = CornerDetector()

    def scan(self, image_path):
        """Hàm thực thi toàn bộ pipeline nhận diện biên 4 góc"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Lỗi: Không thể đọc ảnh {image_path}")
            return None, None, None

        # Thu nhỏ ảnh để tăng tốc độ xử lý mà vẫn giữ được đặc trưng
        ratio = img.shape[0] / 500.0
        orig = img.copy()
        img_resized = cv2.resize(img, (int(img.shape[1] / ratio), 500))

        # Bước 1: Grayscale & Blur
        processed_img = self.preprocessor.process(img_resized)

        # Bước 2: Canny Edge Detection
        edged_img = self.edge_detector.detect(processed_img)

        # Bước 3: Tìm 4 góc
        corners = self.corner_detector.find_corners(edged_img)

        if corners is not None:
            # Vẽ 4 góc lên ảnh gốc (nhân bù lại tỷ lệ)
            cv2.drawContours(orig, [np.int32(corners * ratio)], -1, (0, 255, 0), 3)
            return orig, edged_img, corners
        else:
            return None, edged_img, None
