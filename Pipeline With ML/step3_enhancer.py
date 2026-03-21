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
        
        print("[Step 3] ✒️ Binarize thông minh: Nối nét & Xóa thấm mặt sau...")
        final_img = self.smart_binarize(normalized_img)
        if save_prefix is not None:
            cv2.imwrite(f"{save_prefix}_step3_4_final_enhanced.jpg", final_img)
        
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

    def smart_binarize(self, gray_image):
        """
        [ĐÃ SỬA THEO YÊU CẦU: BỎ XÓA THẤM MẶT SAU]
        Bỏ phương pháp Adaptive Threshold (bộ phận cắt gọt từng vùng nhỏ để xóa thấm, làm mòn chữ lờ mờ).
        Sử dụng phương pháp Global Otsu Thresholding lấy 1 ngưỡng đen trắng chung do phông nền đã được làm trắng đều từ bước Shadow Normalization (3.3).
        """
        # Dùng Otsu binarization
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        return binary

if __name__ == "__main__":
    print("Test Enhancement...")
