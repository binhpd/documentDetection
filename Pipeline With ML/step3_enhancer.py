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

    def enhance(self, image, save_prefix=None, mode="color"):
        """
        Thực hiện toàn bộ pipeline khôi phục Đỉnh Cao với Adaptive Algorithms.
        Trích xuất 2 bản: Đen Trắng (B/W) và Màu (Color)
        """
        # (Đã tắt theo yêu cầu: Bỏ tính năng nắn xô lệch dòng chữ)
        # print("[Step 3] 📐 Đang kiểm tra và nắn thẳng dòng chữ (Deskew & Crop)...")
        # image = self.deskew_and_crop(image)
        # if save_prefix is not None:
        #     cv2.imwrite(f"{save_prefix}_step3_0_deskewed.jpg", image)

        print("[Step 3] 💡 Đang chẩn đoán Flash chói lóa (Glare)...")
        deglared_img = self.remove_glare(image)
        if save_prefix is not None:
            cv2.imwrite(f"{save_prefix}_step3_1_deglared.jpg", deglared_img)
        
        # 1. Pipeline Đen Trắng
        print("[Step 3 - B/W] Đang xử lý bản Đen Trắng Tối Ưu (Adaptive B/W)...")
        bw_final = self.enhance_bw_adaptive(deglared_img)
        if save_prefix is not None:
            cv2.imwrite(f"{save_prefix}_step3_final_bw.jpg", bw_final)
            
        # 2. Pipeline Màu
        print("[Step 3 - Color] Đang xử lý bản Màu Tối Ưu (Adaptive Color)...")
        color_final = self.enhance_color_adaptive(deglared_img)
        if save_prefix is not None:
            cv2.imwrite(f"{save_prefix}_step3_final_color.jpg", color_final)
            
        if mode == "bw":
            return bw_final
        return color_final

    def deskew_and_crop(self, image):
        """
        Nắn thẳng lề chữ chạy lộn xộn (Deskew) và Cắt chém viền dư bừa bãi (Auto Crop)
        """
        # --- 1. Deskew ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Chữ đen nền trắng -> Chữ trắng nền đen
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Chỉ xoay nếu góc lệnh đủ lớn nhưng không phải xoay chọc trời
            if 0.2 < abs(angle) < 15:
                print(f"  -> Thực hiện Deskew xoay {angle:.2f} độ để thẳng dòng cột.")
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))
                
                M[0, 2] += (nW / 2) - center[0]
                M[1, 2] += (nH / 2) - center[1]
                
                image = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

        # --- 2. Auto Crop ---
        gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Lọc ra tất cả pixel không phải màu trắng tinh 255
        _, mask2 = cv2.threshold(gray2, 254, 255, cv2.THRESH_BINARY_INV)
        
        coords2 = np.column_stack(np.where(mask2 > 0))
        if len(coords2) > 0:
            y_min, x_min = np.min(coords2, axis=0)
            y_max, x_max = np.max(coords2, axis=0)
            
            # Đệm lề nhẹ cho đẹp
            pad = 15
            y_min = max(0, y_min - pad)
            x_min = max(0, x_min - pad)
            y_max = min(image.shape[0], y_max + pad)
            x_max = min(image.shape[1], x_max + pad)
            
            image = image[y_min:y_max, x_min:x_max]
            print(f"  -> Cắt viền thừa tự động (Auto Cropped). Kích thước mới: {image.shape[1]}x{image.shape[0]}")
            
        return image

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
        Khử bóng râm đen trắng truyền thống.
        """
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

    def remove_shadows_division_color(self, image):
        """
        Khử bóng râm nhưng giữ nguyên 3 KÊNH MÀU RGB, hoàn toàn khắc phục lỗi Viền Xám và Ám Tím (Colored Halos).
        """
        # Tách màu độc lập B, G, R
        planes = cv2.split(image)
        bg_planes = []
        
        # Kernel siêu nhỏ vì ta sẽ làm việc trên ảnh thu nhỏ (Hiệu năng cực cao)
        kernel_size = 11 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        for plane in planes:
            # 1. Downscale kích thước ảnh xuống 5 lần để tăng tốc nội suy & nới rộng góc nhìn nội suy Bóng Râm
            small_plane = cv2.resize(plane, (0, 0), fx=0.2, fy=0.2)
            
            # 2. Nuốt chửng toàn bộ chi tiết tối mảnh (chữ, nét vẽ) bằng Dilate (Phình to phần giấy trắng)
            bg_small = cv2.dilate(small_plane, kernel)
            
            # 3. Phá hủy răng cưa vuông vắn do hệ lụy của Dilation
            bg_small = cv2.medianBlur(bg_small, 11)
            bg_small = cv2.GaussianBlur(bg_small, (21, 21), 0)
            
            # 4. Phóng to trả lại nguyên bản và dội thêm 1 lần Gaussian lớn nhất để biến mảng Pixel thành dạng Lưới Điện Toả Sáng
            bg_plane = cv2.resize(bg_small, (plane.shape[1], plane.shape[0]))
            bg_plane = cv2.GaussianBlur(bg_plane, (31, 31), 0)
            
            bg_planes.append(bg_plane)
        
        # Ghép rập khuôn 3 lưới điện lại (Ánh sáng thực với đầy đủ nhiệt độ màu Ambient)
        bg_3d = cv2.merge(bg_planes)
        
        # Phép chia "Shadow Division" theo thuyết Nguồn Sáng Kép (Illumination Theory)
        # Nguồn sáng / Background -> Trắng (255) hoàn đối mọi dải màu dù bóng tím đen
        normalized_float = 255.0 * (image.astype(float) / (bg_3d.astype(float) + 1e-6))
        normalized_color = np.clip(normalized_float, 0, 255).astype(np.uint8)
        
        return normalized_color

    def enhance_bw_adaptive(self, deglared_img):
        """
        Pipeline Đen Trắng Tối Ưu bằng Adaptive Threshold & CLAHE
        """
        # 1. Khử bóng bằng remove_shadows_division
        noshadow = self.remove_shadows_division(deglared_img)
        
        # 2. Khử nhiễu cục bộ: bilateralFilter mượt nền giấy nhưng giữ viền chữ rất tốt
        smooth = cv2.bilateralFilter(noshadow, d=5, sigmaColor=50, sigmaSpace=50)
        
        # 3. CLAHE để đẩy mạnh tương phản cục bộ chữ mờ
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(smooth)

        # 4. Phơi sáng mềm tuyến tính kẹp đầu đuôi (Adaptive Percentile)
        p_low, p_high = np.percentile(clahe_img, (2, 90)) # 2% tối nhất thành Đen, 10% sáng nhất thành Trắng
        
        # Đảm bảo khoảng cách logic
        if p_high - p_low < 10:
            p_high = min(255, p_low + 10)
            
        stretched = (clahe_img.astype(np.float32) - p_low) * (255.0 / (p_high - p_low))
        soft_bin = np.clip(stretched, 0, 255).astype(np.uint8)
        
        # 5. Nối đứt gãy bằng Morphology Close 2x2
        inv = cv2.bitwise_not(soft_bin)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        inv_connected = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
        bw_connected = cv2.bitwise_not(inv_connected)
        
        # 6. Làm viền chống rung
        final_bw = self.unsharp_mask(bw_connected)
        return final_bw

    def enhance_color_adaptive(self, deglared_img):
        """
        Pipeline Bản Màu Tối Ưu bằng LAB Denoising và Histogram Auto-Stretching
        """
        # 1. Khử bóng độc lập RGB
        color_noshadow = self.remove_shadows_division_color(deglared_img)
        
        # 2. Khử hạt nhiễu màu trên vùng tối kéo sáng (LAB Denoising)
        lab = cv2.cvtColor(color_noshadow, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Lọc cực gắt Blur trên kênh màu A và B (Khử nhiễu tím/xanh/vàng)
        a = cv2.GaussianBlur(a, (5, 5), 0)
        b = cv2.GaussianBlur(b, (5, 5), 0)
        # Kênh L (độ sáng) giữ nguyên độ sắc nét của viền chữ
        lab_denoised = cv2.merge([l, a, b])
        color_denoised = cv2.cvtColor(lab_denoised, cv2.COLOR_LAB2BGR)
        
        # 3. Adaptive Histogram Stretching (Mềm) tính trên Grayscale cho chuẩn sáng
        gray_for_stats = cv2.cvtColor(color_denoised, cv2.COLOR_BGR2GRAY)
        p_low, p_high = np.percentile(gray_for_stats, (1, 95))
        
        if p_high - p_low < 10:
            p_high = min(255, p_low + 10)
            
        stretched_color = (color_denoised.astype(np.float32) - p_low) * (255.0 / (p_high - p_low))
        stretched_color = np.clip(stretched_color, 0, 255).astype(np.uint8)
        
        # 4. Tăng bão hòa thông minh (Chỉ tăng pixel có màu, không tăng giấy trắng)
        hsv = cv2.cvtColor(stretched_color, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Tăng Saturation 30% ở những chỗ S > 15 (Giấy trắng xám thường có S rất thấp < 10)
        mask_color = s > 15
        s_float = s.astype(np.float32)
        s_float[mask_color] = s_float[mask_color] * 1.3
        s = np.clip(s_float, 0, 255).astype(np.uint8)
        
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced_bgr = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 5. Chốt hạ: Viền chống rung
        final_color = self.unsharp_mask(enhanced_bgr)
        
        return final_color

if __name__ == "__main__":
    print("Test Enhancement...")
