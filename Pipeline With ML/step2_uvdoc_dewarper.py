import cv2
import numpy as np
import os
import torch
import sys

# Thêm đường dẫn tới thư mục UVDoc_repo để import được model và utils
UVDOC_PATH = os.path.join(os.path.dirname(__file__), "UVDoc_repo")
if UVDOC_PATH not in sys.path:
    sys.path.append(UVDOC_PATH)

try:
    from utils import IMG_SIZE, bilinear_unwarping, load_model
    UVDOC_AVAILABLE = True
except ImportError as e:
    UVDOC_AVAILABLE = False
    print(f"❌ [UVDocDewarper] Lỗi import UVDoc_repo: {e}")

class UVDocDewarper:
    def __init__(self, fallback_transformer=None):
        self.fallback = fallback_transformer
        self.is_loaded = False
        
        if not UVDOC_AVAILABLE:
            print("[UVDocDewarper] Thư viện UVDoc không sẵn sàng.")
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model_path = os.path.join(UVDOC_PATH, "model", "best_model.pkl")
        
        if not os.path.exists(self.model_path):
            print(f"[UVDocDewarper] Không tìm thấy file trọng số {self.model_path}")
            return
            
        print(f"[UVDocDewarper] Đang tải mô hình UVDoc trên thiết bị: {self.device}...")
        try:
            # Ở bản Mac MPS, có thể hàm grid_sample / một số kernel CHƯA THỰC SỰ HỖ TRỢ, 
            # tuy nhiên UVDoc utils có gọi nó. Ta cứ load tạm vào self.device
            self.model = load_model(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print("[UVDocDewarper] ✅ Tải mô hình thành công.")
        except Exception as e:
            print(f"❌ [UVDocDewarper] Lỗi khởi tạo mô hình: {e}")
            if self.device.type == "mps":
                print("⚠️ Thử lùi về CPU do lỗi tương thích MPS...")
                self.device = torch.device("cpu")
                try:
                    self.model = load_model(self.model_path)
                    self.model.to(self.device)
                    self.model.eval()
                    self.is_loaded = True
                    print("[UVDocDewarper] ✅ Tải mô hình (CPU) thành công.")
                except Exception as e2:
                    print(f"❌ Lỗi lùi về CPU: {e2}")

    def dewarp(self, img_cropped, save_prefix=None):
        """
        Nắn phẳng tài liệu bằng lưới (Neural Grid-based Document Unwarping).
        
        Args:
            img_cropped: Ảnh đã crop theo bounding box xung quanh tài liệu (bắt buộc, 
                         vì UVDoc resize về 488x712 nên nếu ảnh rộng sẽ bị sai).
                         
        Returns:
            unwarped: BGR numpy array
        """
        if not self.is_loaded:
            print("⚠️ [UVDocDewarper] Model không khả dụng, trả về ảnh gốc.")
            return img_cropped
            
        print("[UVDocDewarper] 🧠 Đang suy luận lưới nắn phẳng (Neural Grid)...")
        try:
            if save_prefix is not None: cv2.imwrite(f"{save_prefix}_step2_uvdoc_1a_cropped.jpg", img_cropped)
            
            # 1. Chuyển đổi định dạng cho PyTorch: RGB, normalize [0, 1]
            # UVDoc utils cần kích thước (1, 3, 488, 712)
            img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # 2. Resize kích thước ảnh về IMG_SIZE để đẩy qua mạng
            inp_resized = cv2.resize(img_rgb, tuple(IMG_SIZE))
            if save_prefix is not None: cv2.imwrite(f"{save_prefix}_step2_uvdoc_1b_resized.jpg", (inp_resized * 255).astype(np.uint8)[:,:,::-1])
            # HWC -> CHW, tạo batch dim
            inp_tensor = torch.from_numpy(inp_resized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

            # 3. Model Inference: Lấy point_positions2D (Lưới cong dạng [1, 2, Gh, Gw])
            with torch.no_grad():
                point_positions2D, _ = self.model(inp_tensor)

            # 4. Unwarping dùng bilinear interpolation
            original_size = img_rgb.shape[:2][::-1] # (Width, Height) của lưới ảnh cần unwarp
            
            # warped_img đầu vào cũng cần convert -> TENSORS dạng BxCxHxW
            img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            
            # Do PyTorch grid_sample trên MPS đang lỗi (Tính đến torch 2.1+) ,
            # Cẩn thận catch nếu bilinear_unwarping chết
            try:
                unwarped_tensor = bilinear_unwarping(
                    warped_img=img_tensor,
                    point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
                    img_size=original_size,
                )
            except Exception as e_grid:
                if self.device.type == "mps" and "grid_sampler" in str(e_grid).lower():
                    # Fallback CPU for grid sample
                    print("[UVDoc] Cảnh báo lỗi `grid_sample` trên MPS, đổi qua CPU nội suy...")
                    unwarped_tensor = bilinear_unwarping(
                        warped_img=img_tensor.cpu(),
                        point_positions=torch.unsqueeze(point_positions2D[0].cpu(), dim=0),
                        img_size=original_size,
                    )
                else:
                    raise e_grid
            
            # 5. Đưa trả lại OpenCV Numpy array
            unwarped = (unwarped_tensor[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            unwarped_bgr = cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR)

            print("[UVDocDewarper] ✅ Nắn phẳng UVDoc thành công!")
            return unwarped_bgr
            
        except Exception as e:
            print(f"❌ [UVDocDewarper] Lỗi thực thi nắn phẳng: {e}")
            return img_cropped
