import cv2
import numpy as np

class DocumentEnhancer:
    """
    Step 3: Khôi phục và Tăng cường chất lượng tài liệu (CV Enhancement).
    Nhận vào ảnh đã được Warp Perspective (vuông vắn nhưng có thể bị tối/bóng đổ).
    Trả ra ảnh sáng đẹp, chuẩn cấu trúc Scanner App.
    """
    def __init__(self):
        pass

    def enhance(self, image, save_prefix=None):
        """
        Thực hiện toàn bộ pipeline khôi phục Đỉnh Cao (Mới cập nhật 4 chốt chặn):
        0. Khôi phục lóa sáng Flash (In-painting)
        1. Chống Rung Nhoè camera tay (Unsharp Masking)
        2. Khử bóng & Thấm mực mặt sau (Illumination Normalization)
        3. Tăng cường chi tiết, nối nét chữ đứt gãy (Morphology)
        """
        print("[Step 3] 💡 Đang chẩn đoán Flash chói lóa (Glare)...")
        deglared_img = self.remove_glare(image)
        if save_prefix is not None:
            cv2.imwrite(f"{save_prefix}_step3_1_deglared.jpg", deglared_img)
        
        print("[Step 3] 🔍 Kéo gắt biên cạnh chống rung mờ (Deblurring)...")
        sharpened_img = self.unsharp_mask(deglared_img)
        if save_prefix is not None:
            cv2.imwrite(f"{save_prefix}_step3_2_sharpened.jpg", sharpened_img)
        
        print("[Step 3] ☁️ Khử loang lổ bóng râm (Shadow Normalization)...")
        normalized_img = self.remove_shadows_division(sharpened_img)
        if save_prefix is not None:
            cv2.imwrite(f"{save_prefix}_step3_3_noshadow.jpg", normalized_img)
        
        print("[Step 3] ✒️ Binarize thông minh (Phơi sáng mềm)... đang tạo 5 mức độ để lựa chọn")
        
        # 5 Cặp tham số tương quan (Black Point, White Point)
        threshold_pairs = [
            (90, 220),   # Option 1: Rất mềm mại (Nhiều hạt xám, như bút chì nét nhạt)
            (110, 200),  # Option 2: Cân bằng tiêu chuẩn (Đủ đen gắt, giữ viền mượt)
            (130, 190),  # Option 3: Chữ mạnh, nền trắng phau
            (160, 180),  # Option 4: Cực gắt (Chữ đen đậm thui, lằn ranh xám mỏng, gần như B/W cũ)
            (70, 150)    # Option 5: Phơi sáng chói (Tẩy trắng nền cực mạnh, giữ lại chữ vừa in)
        ]
        
        final_img = None
        for i, (bp, wp) in enumerate(threshold_pairs, 1):
            soft_bin = self.smart_binarize(normalized_img, black_point=bp, white_point=wp)
            # Nếu chạy trên app có lưu cache ảnh 
            if save_prefix is not None:
                filename = f"{save_prefix}_step3_4_opt{i}_B{bp}_W{wp}.jpg"
                cv2.imwrite(filename, soft_bin)
                print(f"  -> Lưu mẫu thử Option {i} (Đen:{bp}, Trắng:{wp}) ra {filename}")
            
            # Tạm khóa Option 2 làm chuẩn bị mặc định trả về cho Matplotlib show()
            if i == 2: 
                final_img = soft_bin
                
        return final_img

    def remove_glare(self, image):
        """
        Khắc phục Vấn đề 4: Lóa sáng Flash (Glare/Reflection).
        Dùng hệ thống In-painting của OpenCV để đoán và vá lại vùng giấy trắng xóa.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Giới hạn ngưỡng những điểm lóe trắng lóa mắt. 
        # (Nâng cực gắt lên 250 để tránh nhầm lẫn tờ giấy trắng phau thông thường)
        _, glare_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)
        
        # Chỉ Inpaint nếu diện tích lóa flash cực nhỏ (nhỏ hơn 5% diện tích ảnh). 
        # Nếu lớn hơn, chắc chắn đó là màu giấy trắng tự nhiên -> Không được nội suy làm mù chữ!
        total_pixels = gray.shape[0] * gray.shape[1]
        if 0 < cv2.countNonZero(glare_mask) < (total_pixels * 0.05):
            return cv2.inpaint(image, glare_mask, 3, cv2.INPAINT_TELEA)
        return image

    def unsharp_mask(self, image):
        """
        Khắc phục Vấn đề 1: Rung nhòe tay (Motion Blur).
        Sử dụng Unsharp Masking kéo tần số cao làm mép chữ sắc như dao cạo.
        """
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return unsharp

    def remove_shadows_division(self, image):
        """
        Phương pháp "Division-based Illumination Normalization".
        Dùng để khử cực sạch các bóng đen/ngón tay ám trên giấy.
        """
        # Chuyển xám để lấy bản đồ ánh sáng (nếu chưa xám)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        kernel_size = 21
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        bg_estimate = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        bg_estimate = cv2.GaussianBlur(bg_estimate, (51, 51), 0)
        
        normalized_float = 255.0 * (gray.astype(float) / (bg_estimate.astype(float) + 1e-6))
        normalized_gray = np.clip(normalized_float, 0, 255).astype(np.uint8)
        
        return normalized_gray

    def smart_binarize(self, gray_image, black_point=110.0, white_point=200.0):
        """
        [ĐÃ SỬA THEO YÊU CẦU: SOFT BINARIZATION]
        Giữ lại các pixel xám mượt ở biên chữ (anti-aliasing) để tránh gai vỡ hay đứt chữ mảnh.
        Ép vùng lõi chữ thành Đen nhánh và nền thành Trắng tinh dựa trên Point Thresholds.
        """
        # Xác định điểm Đen (Black Point) và điểm Trắng (White Point)
        # Pixel <= black_point sẽ chuyển thành 0 (Đen tuyệt đối)
        # Pixel >= white_point sẽ chuyển thành 255 (Trắng tuyệt đối)
        # Vùng ở giữa biến thành xám trung gian làm mượt rìa chữ.
        
        # Phép chiếu biến đổi tuyến tính (Linear Contrast Stretching)
        stretched = (gray_image.astype(np.float32) - float(black_point)) * (255.0 / float(white_point - black_point))
        
        # Cắt gọt chuẩn hóa kết quả vào dải uint8 (0-255)
        soft_binary = np.clip(stretched, 0, 255).astype(np.uint8)
        
        return soft_binary

if __name__ == "__main__":
    print("Test Enhancement...")
