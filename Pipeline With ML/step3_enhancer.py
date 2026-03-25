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
        # (Đã tắt theo yêu cầu: Bỏ tính năng nắn xô lệch dòng chữ)
        # print("[Step 3] 📐 Đang kiểm tra và nắn thẳng dòng chữ (Deskew & Crop)...")
        # image = self.deskew_and_crop(image)
        # if save_prefix is not None:
        #     cv2.imwrite(f"{save_prefix}_step3_0_deskewed.jpg", image)

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
                
        # --- TÍNH NĂNG MỚI: XỬ LÝ ẢNH MÀU NGUYÊN BẢN ---
        print("[Step 3] 🎨 Tái tạo ảnh Màu sắc (Color Mode)... Tẩy trắng nền giữ nguyên màu sắc chữ, dấu mộc")
        color_noshadow = self.remove_shadows_division_color(sharpened_img)
        
        # 5 Cặp tham số tương quan cho CHẾ ĐỘ ẢNH MÀU
        color_threshold_pairs = [
            (30, 240),  # Option 6: Giữ Trọn Vẹn Cả Màu Siêu Nhạt (Bút highlight nhạt, phấn màu, logo chìm)
            (40, 230),  # Option 7: Cân Bằng Chuẩn Màu Sắc (Trắng tinh khôi, màu dịu mắt tự nhiên)
            (50, 210),  # Option 8: Dấu Mộc Đậm Hơn, Nền Trắng Sáng
            (60, 190),  # Option 9: Gắt - Chữ đen nhánh & Mộc Đỏ Chót, Nền Cháy Sáng
            (70, 170)   # Option 10: Siêu Gắt - Ép mọi màu nhạt rực rỡ lên tối đa
        ]
        
        best_color_enhanced = None
        for i, (bp, wp) in enumerate(color_threshold_pairs, 6):
            color_enhanced = self.enhance_color(color_noshadow, bp=float(bp), wp=float(wp))
            
            if save_prefix is not None:
                filename = f"{save_prefix}_step3_4_opt{i}_Color_B{bp}_W{wp}.jpg"
                cv2.imwrite(filename, color_enhanced)
                print(f"  -> Lưu mẫu thử Ảnh Màu Opt {i} (Đen:{bp}, Trắng:{wp}) ra {filename}")
                
            if i == 7: # Option 7 (40-230) là mức tiêu chuẩn vàng, ta lấy làm đại diện nếu muốn return
                best_color_enhanced = color_enhanced
            
        # Có thể return best_color_enhanced nếu muốn show bản màu lên cuối cùng, hoặc return final_img
        return best_color_enhanced

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

    def enhance_color(self, normalized_color, bp=40.0, wp=230.0):
        """
        Thổi bay bóng râm mờ còn sót lại và nâng mức bão hòa màu sắc.
        """
        # 1. Soft Binarization TRÊN CẢ 3 KÊNH MÀU
        # (CẬP NHẬT NGƯỠNG RẤT NHẸ): Do đã khử bóng triệt để ở bước trên, nền giấy lúc này rất gần 255.
        # Chúng ta chỉ nên ép nhẹ những phổ xám > wp thành Trắng tinh, và kéo màu nhạt xuống xíu (bp).
        # Tuyệt đối không để wp quá thấp (185) trừ phi cố tình muốn gắt màu.
        
        stretched = (normalized_color.astype(np.float32) - bp) * (255.0 / (wp - bp))
        stretched_color = np.clip(stretched, 0, 255).astype(np.uint8)
        
        # 2. Chuyển sang không gian màu HSV để kích bão hòa (Đâm chồi màu sắc)
        hsv = cv2.cvtColor(stretched_color, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # Tăng Saturation (độ rực rỡ) lên 30% để mực nét hơn, con dấu đỏ chót hơn
        s = s * 1.3
        s = np.clip(s, 0, 255)
        
        hsv_enhanced = cv2.merge([h, s, v]).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 3. Áp dụng bộ lọc Unsharp Masking 1 lần nữa ở layer cuối cùng để sắc bén
        gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.0)
        final_sharp = cv2.addWeighted(enhanced_bgr, 1.2, gaussian, -0.2, 0)
        
        return final_sharp

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
