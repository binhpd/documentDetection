import cv2

class EdgeDetector:
    def __init__(self, min_val=75, max_val=200):
        self.min_val = min_val
        self.max_val = max_val

    def detect(self, image):
        """Phát hiện biên bằng thuật toán Canny"""
        edged = cv2.Canny(image, self.min_val, self.max_val)
        return edged
