import cv2

class ImagePreprocessor:
    def __init__(self, blur_kernel_size=(5, 5)):
        self.blur_kernel_size = blur_kernel_size

    def process(self, image):
        """Chuyển ảnh sang xám và làm mờ để khử nhiễu"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)
        return blurred
