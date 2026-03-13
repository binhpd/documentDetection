"""
Step 1: DocAligner Document Segmentation
🔴 Machine Learning (State-of-the-Art)

Chức năng:
- Sử dụng thư viện DocAligner (DocsaidLab) để tìm đúng viền và 4 góc tài liệu.
- Thay thế hoàn toàn YOLOv8 vì nó bao đường viền cực khít và chuyên dụng hơn cho bài toán căn lề tài liệu.

Input:  Ảnh BGR (H × W × 3, uint8)
Output: Mask nhị phân (H × W × 1, 0/255) + 4 toạ độ góc (4, 2) float32
"""
import cv2
import numpy as np
import os

class DocAlignerSegmentor:
    def __init__(self):
        self.is_loaded = False
        try:
            # Sửa lỗi thư viện C++ TurboJPEG khác architecture trên Mac Apple Silicon
            os.environ["TURBOJPEG"] = "/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.dylib"
            
            from docaligner import DocAligner
            import torch
            
            # Cấu hình thiết bị backend (MPS cho Mac Silicon, hoặc CPU/CUDA)
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                
            print(f"[DocAligner] Khởi tạo mô hình trên thiết bị thử nghiệm: {device}")
            # Khởi tạo mô hình. Nó sẽ tự động tải weights (ONNX/Torch) xuống bộ nhớ đệm
            try:
                self.model = DocAligner(backend=device)
            except Exception as backend_err:
                if device == "mps":
                    print(f"⚠️ [DocAligner] Lỗi backend {device} (Thường do onnxruntime chưa hỗ trợ CoreML/MPS). Tự động lùi về CPU...")
                    device = "cpu"
                    self.model = DocAligner(backend=device)
                else:
                    raise backend_err
            
            self.is_loaded = True
        except ImportError as e:
            print(f"❌ [DocAligner] Chưa cài đặt docaligner hoặc thiếu dependency: {e}")
        except Exception as e:
            print(f"❌ [DocAligner] Lỗi khởi tạo model: {e}")

    def segment(self, image):
        if not self.is_loaded:
            return None, None
            
        print("[ML] Chạy phân tích DocAligner inference...")
        
        try:
            # Chạy inference
            # Model yêu cầu input là ảnh BGR array
            polygon_pts = self.model(image)
            
            # Nếu không tìm thấy, model trả về None hoặc list rỗng
            if polygon_pts is None or len(polygon_pts) == 0:
                print("❌ [DocAligner] Không tìm thấy tài liệu!")
                return None, None
            
            # polygon_pts là ma trận numpy (4, 2) theo thứ tự viền.
            print(f"  ✓ DocAligner tìm thấy 4 góc tài liệu!")
            
            # Tạo mask giả lập để tương thích với pipeline cũ Step 3
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_pts.astype(np.int32)], 255)
            
            return mask, polygon_pts.astype(np.float32)
            
        except Exception as e:
            print(f"❌ [DocAligner] Lỗi trong quá trình predict: {e}")
            return None, None
