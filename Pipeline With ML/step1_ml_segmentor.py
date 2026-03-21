"""
Step 1e: Fallback cuối — U-Net Document Segmentation
🔴 Machine Learning

Chức năng:
- Khi cả approxPolyDP (1c) và Hough Lines (1d) đều thất bại
- Dùng mạng U-Net đã huấn luyện để phân vùng ngữ nghĩa: pixel nào là "giấy", pixel nào là "nền"
- Từ mask nhị phân → trích xuất 4 góc bằng minAreaRect hoặc Hough Lines

Input:  Ảnh màu đã resize (H × W × 3, uint8)
Output: Mask nhị phân (H × W × 1, 0/255) + 4 toạ độ góc (4, 2) float32

═══════════════════════════════════════════════════════════════════
HƯỚNG DẪN SỬ DỤNG U-Net CHO DOCUMENT SEGMENTATION
═══════════════════════════════════════════════════════════════════

1. U-Net là gì?
   - Kiến trúc mạng nơ-ron dạng Encoder-Decoder
   - Encoder: Thu nhỏ ảnh dần dần, học các đặc trưng (feature) trừu tượng
   - Decoder: Phóng to lại, kết hợp feature từ encoder (skip connections)
   - Output: 1 ảnh mask cùng kích thước input, mỗi pixel = xác suất thuộc "giấy"

2. Cần gì để sử dụng?
   - pip install torch torchvision  (PyTorch framework)
   - Hoặc: pip install ultralytics  (YOLOv8 — đơn giản hơn rất nhiều)
   - File model (.pt / .onnx) đã huấn luyện

3. Hai lựa chọn:
   a) YOLOv8-Seg (KHUYÊN DÙNG — đơn giản nhất):
      - pip install ultralytics
      - Chỉ cần 3 dòng code để chạy inference
      - Có thể dùng pre-trained model hoặc fine-tune trên dataset tài liệu

   b) U-Net tự xây (Học thuật — phù hợp trình bày đồ án):
      - Tự định nghĩa kiến trúc Encoder-Decoder
      - Huấn luyện trên dataset tài liệu (SmartDoc-QA, MIDV-500)
      - Hiểu sâu hơn về cách mạng hoạt động
"""

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════
# PHƯƠNG ÁN 1: YOLOv8-Seg (Đơn giản nhất, khuyên dùng)
# ═══════════════════════════════════════════════════════════════

class YOLOSegmentor:
    """Document segmentation sử dụng YOLOv8-Seg.

    Sử dụng pre-trained model trên COCO (80 classes).
    Chiến lược tìm tài liệu:
        1. Tìm object thuộc class 'book' (class_id=73) → khả năng cao là tài liệu
        2. Nếu không có 'book', lấy mask lớn nhất trong ảnh
        3. Từ mask → minAreaRect → 4 góc

    Model sẽ tự động download lần đầu (~6MB cho yolov8n-seg.pt).
    """

    # COCO classes có liên quan đến tài liệu
    DOCUMENT_CLASSES = {
        73: 'book',
        67: 'cell phone',  # Đôi khi tài liệu bị nhận nhầm
    }

    def __init__(self, model_path="yolov8n-seg.pt"):
        self.model_path = model_path
        self.model = None

    def _load_model(self):
        """Lazy load model (chỉ load khi cần, tự download nếu chưa có)."""
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                print(f"[ML] Đã load YOLO model: {self.model_path}")
            except ImportError:
                print("[ML] Chưa cài ultralytics. Chạy: pip install ultralytics")
                return False
            except Exception as e:
                print(f"[ML] Lỗi load model: {e}")
                return False
        return True

    def segment(self, image):
        """Phân vùng tài liệu bằng YOLO.

        Chiến lược:
            1. Chạy YOLO inference → danh sách masks + classes
            2. Ưu tiên mask thuộc class 'book'
            3. Nếu không có → lấy mask có diện tích lớn nhất

        Returns:
            mask: Ảnh nhị phân (H × W), 255 = giấy, 0 = nền
            corners: np.array (4, 2) float32 — hoặc None
        """
        if not self._load_model():
            return None, None

        results = self.model(image, verbose=False)

        if len(results) == 0 or results[0].masks is None:
            print("[ML] YOLO không phát hiện object nào")
            return None, None

        masks = results[0].masks.data.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        print(f"[ML] YOLO phát hiện {len(masks)} object(s):")
        for i, (cls_id, conf) in enumerate(zip(classes, confs)):
            cls_name = results[0].names.get(cls_id, f"class_{cls_id}")
            print(f"      #{i}: {cls_name} (confidence={conf:.2f})")

        # Chiến lược 1: Tìm mask thuộc class 'book'
        best_mask = None
        best_score = -1

        for i, cls_id in enumerate(classes):
            if cls_id in self.DOCUMENT_CLASSES:
                score = confs[i]
                if score > best_score:
                    best_score = score
                    best_mask = masks[i]
                    print(f"[ML] → Chọn #{i} ({self.DOCUMENT_CLASSES[cls_id]}) conf={score:.2f}")

        # Chiến lược 2: Fallback → lấy mask lớn nhất (Nhưng loại trừ 'person' - class 0)
        if best_mask is None:
            # Lọc bỏ class 0 ('person') vì YOLO rất hay nhận diện nhầm tay người cầm giấy thành vật thể chính
            valid_indices = [idx for idx, cls_id in enumerate(classes) if cls_id != 0]
            
            if len(valid_indices) > 0:
                areas = [masks[idx].sum() for idx in valid_indices]
                best_valid_idx = valid_indices[int(np.argmax(areas))]
                best_mask = masks[best_valid_idx]
                cls_name = results[0].names.get(classes[best_valid_idx], "unknown")
                print(f"[ML] → Không tìm thấy 'book'. Dùng mask hợp lệ lớn nhất: #{best_valid_idx} ({cls_name})")
            else:
                print(f"[ML] → Chỉ tìm thấy 'person' (TAY NGƯỜI) trong ảnh. Bỏ qua để tránh cắt nhầm tay.")
                return None, None

        # Chuyển mask thành binary
        mask = (best_mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Trích 4 góc từ mask
        corners = self._mask_to_corners(mask)
        return mask, corners

    def _mask_to_corners(self, mask):
        """Trích xuất 4 góc từ mask nhị phân.

        Sử dụng minAreaRect để tìm hình chữ nhật xoay nhỏ nhất
        bao quanh vùng trắng (giấy) trong mask.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# PHƯƠNG ÁN 2: U-Net tự xây (Phù hợp trình bày đồ án)
# ═══════════════════════════════════════════════════════════════

class SimpleUNet:
    """U-Net đơn giản cho document segmentation.
    
    Kiến trúc:
        Encoder (Thu nhỏ):
            Conv2d(3→64) → ReLU → MaxPool → 
            Conv2d(64→128) → ReLU → MaxPool → 
            Conv2d(128→256) → ReLU → MaxPool
        
        Bottleneck:
            Conv2d(256→512) → ReLU
        
        Decoder (Phóng to):
            Upsample → Conv2d(512+256→256) → ReLU →
            Upsample → Conv2d(256+128→128) → ReLU →
            Upsample → Conv2d(128+64→64) → ReLU →
            Conv2d(64→1) → Sigmoid
        
        Skip Connections: Nối output encoder với input decoder tương ứng
        
    Để sử dụng:
        1. pip install torch torchvision
        2. Huấn luyện: xem hàm train_example() bên dưới
        3. Inference: xem hàm predict()
    """

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.device = None

    def build_model(self):
        """Xây dựng kiến trúc U-Net bằng PyTorch."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            print("[ML] Chưa cài PyTorch. Chạy: pip install torch torchvision")
            return None

        class UNetBlock(nn.Module):
            """1 block: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU"""
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            def forward(self, x):
                return self.block(x)

        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder
                self.enc1 = UNetBlock(3, 64)
                self.enc2 = UNetBlock(64, 128)
                self.enc3 = UNetBlock(128, 256)
                self.pool = nn.MaxPool2d(2)

                # Bottleneck
                self.bottleneck = UNetBlock(256, 512)

                # Decoder
                self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
                self.dec3 = UNetBlock(512, 256)  # 512 = 256 (up) + 256 (skip)
                self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec2 = UNetBlock(256, 128)
                self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec1 = UNetBlock(128, 64)

                # Output
                self.out_conv = nn.Conv2d(64, 1, 1)

            def forward(self, x):
                # Encoder
                e1 = self.enc1(x)                    # (B, 64, H, W)
                e2 = self.enc2(self.pool(e1))        # (B, 128, H/2, W/2)
                e3 = self.enc3(self.pool(e2))        # (B, 256, H/4, W/4)

                # Bottleneck
                b = self.bottleneck(self.pool(e3))   # (B, 512, H/8, W/8)

                # Decoder + Skip Connections
                d3 = self.up3(b)                     # (B, 256, H/4, W/4)
                d3 = torch.cat([d3, e3], dim=1)      # (B, 512, H/4, W/4)
                d3 = self.dec3(d3)                   # (B, 256, H/4, W/4)

                d2 = self.up2(d3)
                d2 = torch.cat([d2, e2], dim=1)
                d2 = self.dec2(d2)

                d1 = self.up1(d2)
                d1 = torch.cat([d1, e1], dim=1)
                d1 = self.dec1(d1)

                return torch.sigmoid(self.out_conv(d1))  # (B, 1, H, W)

        return UNet()

    def load_model(self):
        """Load model đã huấn luyện."""
        try:
            import torch
        except ImportError:
            print("[ML] Chưa cài PyTorch.")
            return False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()
        if self.model is None:
            return False

        if self.model_path:
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[ML] Đã load U-Net model: {self.model_path}")
            except Exception as e:
                print(f"[ML] Lỗi load model: {e}")
                return False

        self.model.to(self.device)
        self.model.eval()
        return True

    def predict(self, image, input_size=256):
        """Dự đoán mask tài liệu.
        
        Args:
            image: Ảnh BGR (H × W × 3, uint8)
            input_size: Kích thước input cho mạng (vuông)
            
        Returns:
            mask: Ảnh nhị phân (H × W), 255 = giấy, 0 = nền
            corners: np.array (4, 2) float32 — hoặc None
        """
        import torch

        orig_h, orig_w = image.shape[:2]

        # Tiền xử lý: resize → normalize → tensor
        resized = cv2.resize(image, (input_size, input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0  # (3, H, W)
        tensor = tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # Inference
        with torch.no_grad():
            output = self.model(tensor)  # (1, 1, H, W)
        
        # Hậu xử lý: tensor → mask
        prob_map = output.squeeze().cpu().numpy()  # (H, W) giá trị 0.0-1.0
        mask = (prob_map > 0.5).astype(np.uint8) * 255  # Threshold tại 0.5
        mask = cv2.resize(mask, (orig_w, orig_h))  # Resize về kích thước gốc

        # Trích 4 góc từ mask
        corners = self._mask_to_corners(mask)
        return mask, corners

    def _mask_to_corners(self, mask):
        """Trích xuất 4 góc từ mask (giống YOLOSegmentor)."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# WRAPPER: Gộp cả 2 phương án
# ═══════════════════════════════════════════════════════════════

class MLSegmentor:
    """Wrapper tự động chọn YOLO hoặc U-Net.

    Thứ tự ưu tiên:
        1. YOLO (mặc định dùng yolov8n-seg.pt pre-trained, tự download)
        2. U-Net (nếu có model đã huấn luyện)
    """

    def __init__(self, yolo_model_path=None, unet_model_path=None):
        # Mặc định dùng YOLO pre-trained nếu không chỉ định gì
        self.yolo_model_path = yolo_model_path or "yolov8n-seg.pt"
        self.unet_model_path = unet_model_path

    def segment(self, image):
        """Thử phân vùng bằng ML.

        Returns:
            mask, corners — hoặc (None, None) nếu ML không khả dụng
        """
        # Thử YOLO trước (mặc định luôn có vì dùng pre-trained)
        try:
            segmentor = YOLOSegmentor(self.yolo_model_path)
            mask, corners = segmentor.segment(image)
            if corners is not None:
                print("[ML] ✓ Phân vùng thành công bằng YOLO")
                return mask, corners
        except Exception as e:
            print(f"[ML] YOLO thất bại: {e}")

        # Thử U-Net nếu có model
        if self.unet_model_path:
            try:
                unet = SimpleUNet(self.unet_model_path)
                if unet.load_model():
                    mask, corners = unet.predict(image)
                    if corners is not None:
                        print("[ML] ✓ Phân vùng thành công bằng U-Net")
                        return mask, corners
            except Exception as e:
                print(f"[ML] U-Net thất bại: {e}")

        print("[ML] ✗ Không thể phân vùng bằng ML")
        return None, None
