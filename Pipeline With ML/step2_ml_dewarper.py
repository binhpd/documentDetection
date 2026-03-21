"""
Step 2c: Document Dewarping bằng Machine Learning
🔴 Machine Learning

Chức năng:
- Khi Perspective Transform (2b) không đủ (tài liệu bị cong méo vật lý như trang sách dày, hoặc tờ giấy bị vò nát).
- Sử dụng mô hình Deep Learning (ví dụ: DewarpNet, GeoTr, DocUNet) để dự đoán một bản đồ
  biến đổi phi tuyến tính (Backward Mapping Grid).
- Dùng lưới này để "nặn" (remap) từng pixel của mảnh giấy cong thành một mặt phẳng hình chữ nhật.

LƯU Ý QUAN TRỌNG:
- Các mô hình Dewarping đạt State-of-the-Art hiện nay (như GeoTr) nặng khoảng 160MB - 500MB+ và code setup phức tạp.
- Module này được viết dưới dạng "Mẫu Kiến trúc" (Architecture Blueprint) để tích hợp sẵn vào Pipeline đồ án.
- Nếu không có file model weights (`.pth`), code sẽ tự động Fallback (lùi về) Perspective Transform thông thường.

Input:  
- image: Ảnh BGR đã crop vùng quét (nghiêng/cong)
Output: 
- dewarped_image: Ảnh BGR đã được ủi phẳng toàn bộ độ cong
"""

import cv2
import numpy as np
import os
import tempfile
import sys

try:
    from page_dewarp.image import WarpedImage
    from page_dewarp.options import Config
except ImportError:
    WarpedImage = None

class MLDewarper:
    def __init__(self, fallback_transformer=None):
        self.fallback = fallback_transformer
        self.model_loaded = False
        self._load_model()
        
    def _load_model(self):
        """Kiểm tra package page-dewarp."""
        if WarpedImage is not None:
            self.model_loaded = True
            print(f"[ML Dewarp] ✅ Đã tích hợp thuật toán Text-line Dewarping (page-dewarp) thành công.")
        else:
            print("[ML Dewarp] ❌ Thiếu thư viện page-dewarp. (Run: pip install page-dewarp)")
            self.model_loaded = False
        return self.model_loaded

    def dewarp(self, image, corners=None):
        """
        Là phẳng ảnh tài liệu cong bằng phân tích dòng chữ (page_dewarp).
        
        Args:
            image: Ảnh gốc (numpy array bgr)
            corners: 4 tọa độ góc (dùng để crop ảnh trước khi dewarp)
            
        Returns:
            dewarped_image: Ảnh đã được uốn phẳng
        """
        # Nếu model không tồn tại → chạy Fallback
        if not self.model_loaded:
            if self.fallback and corners is not None:
                return self.fallback.transform(image, corners)
            return image

        # 1. Cắt ảnh (Crop) bằng Perspective Transform trước
        cropped_image = image
        if self.fallback and corners is not None:
             print("[ML Dewarp] ✂️ Đang cắt vùng tài liệu dựa trên 4 góc...")
             cropped_image = self.fallback.transform(image, corners)

        print("[ML Dewarp] 🧠 Đang phân tích độ cong dòng chữ (Text-line Dewarping)...")
        
        # page-dewarp yêu cầu truyền file path cho WarpedImage, ta tạo file tạm thời.
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
                temp_filename = temp_img.name
                
            cv2.imwrite(temp_filename, cropped_image)
            
            # Khởi tạo config cho page-dewarp
            config = Config()
            config.NO_BINARY = 1 # Không xuất ảnh nhị phân vì Pipeline có bước Binarization riêng
            
            # Cấu hình giảm log nếu cần thiết (1 là cảnh báo, 0 là tắt)
            config.DEBUG_LEVEL = 0
            
            # Chạy dewarping
            warped = WarpedImage(temp_filename, config=config)
            
            if warped.written and warped.outfile and os.path.exists(warped.outfile):
                # Đọc ảnh đã được ủi phẳng
                dewarped_image = cv2.imread(str(warped.outfile))
                
                # Dọn dẹp file tạm
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                if os.path.exists(warped.outfile):
                    os.remove(warped.outfile)
                    
                print("[ML Dewarp] ✅ Dewarping hình học dòng chữ thành công!")
                return dewarped_image
            else:
                raise ValueError("Không tạo được ảnh từ page-dewarp")
                
        except Exception as e:
            print(f"[ML Dewarp] Lỗi thực thi Text-line Dewarping: {e}")
            # Nếu thất bại → trả về ảnh đã crop
            return cropped_image

