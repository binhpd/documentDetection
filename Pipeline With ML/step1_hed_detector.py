"""
Step 1b_HED: Phát hiện cạnh bằng Neural Network (HED)
🟢 Image Processing (ML-based)

Chức năng:
- Thay thế Canny Edge Detection (1b) bằng Holistically-Nested Edge Detection (HED).
- HED dùng mạng CNN để học và nhận diện cạnh có ý nghĩa (như mép giấy), bỏ qua nhiễu nền.
- Trả về ảnh nhị phân chứa cạnh tài liệu.

Input:  Ảnh màu hoặc xám (H × W × 3, uint8)
Output: Ảnh nhị phân cạnh (H × W × 1, uint8, giá trị 0 hoặc 255)

Lưu ý:
Cần 2 file cấu hình model trong thư mục `models/`:
1. deploy.prototxt (đã có)
2. hed_pretrained_bsds.caffemodel (nếu chưa có, xem log hướng dẫn tải)
"""

import cv2
import numpy as np
import os


class CropLayer(object):
    """
    HED yêu cầu một custom crop layer trong Caffe.
    Layer này sẽ cắt feature map về cùng kích thước.
    """
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


class HEDDetector:
    def __init__(self, prototxt_path, caffemodel_path):
        """Khởi tạo detector. Load model từ file."""
        self.prototxt_path = prototxt_path
        self.caffemodel_path = caffemodel_path
        self.net = None
        self.is_loaded = False
        
        # Đăng ký custom crop layer cho OpenCV DNN
        cv2.dnn_registerLayer('Crop', CropLayer)

    def _load_model(self):
        """Lazy load model dnn cùa openCV."""
        if self.is_loaded:
            return True

        if not os.path.exists(self.prototxt_path) or not os.path.exists(self.caffemodel_path):
            print(f"❌ [HED] Lỗi: Không tìm thấy file model HED.")
            print(f"   Vui lòng tải model HED (~56MB):")
            print(f"   File prototxt: {self.prototxt_path}")
            print(f"   File caffemodel: {self.caffemodel_path}")
            print(f"   Tải tại: https://github.com/ashukumar27/HED-Edge-Detection/tree/master (hoặc nguồn khác)")
            return False
            
        try:
            # Kiểm tra filesize caffemodel (nếu < 1MB thì là file rác)
            if os.path.getsize(self.caffemodel_path) < 1000000:
                print(f"❌ [HED] File .caffemodel bị lỗi (quá nhỏ). Bạn phải tải thủ công file thật (56MB).")
                return False

            self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.caffemodel_path)
            self.is_loaded = True
            print("[HED] Đã load thành công Holistically-Nested Edge Detection model.")
            return True
        except Exception as e:
            print(f"❌ [HED] Lỗi load model DNN: {e}")
            return False

    def detect(self, image):
        """
        Inference HED để trích xuất cạnh.
        
        Args:
            image: Ảnh BGR đã resize.
        Returns:
            edged: Ảnh nhị phân cạnh, cùng kích thước.
        """
        if not self._load_model():
            return None

        (H, W) = image.shape[:2]

        # Chuyển ảnh BGR sang DNN blob
        # (Mean subtraction: (104.0, 119.0, 122.0) là giá trị training gốc của ImageNet)
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                                     mean=(104.0, 119.0, 122.0),
                                     swapRB=False, crop=False)

        # Forward pass qua mạng
        self.net.setInput(blob)
        hed_output = self.net.forward()
        
        # Chuẩn hoá về [0, 255]
        hed_output = (255 * hed_output).astype("uint8")
        
        # HED sinh ra probability map (viền gộp cả nét dày và mờ).
        # Thay vì filter Canny (làm đứt nét viền ngoài), ta dùng trực tiếp HED.
        # Dùng Otsu threshold để tự động binarize HED map.
        _, hed_mask = cv2.threshold(hed_output, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # HED mask có thể bị thủng lờ mờ vài pixel ở những chỗ quá sáng.
        # Dùng phép toán Đóng (Close) Morphological để hàn gắn các nét viền đứt gãy.
        # Điều này đảm bảo viền tờ giấy tạo thành 1 vòng tròn khép kín hoàn hảo.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed_edges = cv2.morphologyEx(hed_mask, cv2.MORPH_CLOSE, kernel)
        
        # Phép toán làm mảnh (thinner) viền nếu cần (tùy chọn) để Hough Lines chạy tốt hơn.
        # Nhưng approxPolyDP lại rất thích viền dày khép kín này.
        return closed_edges
