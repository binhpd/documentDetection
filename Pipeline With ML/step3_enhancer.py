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

    def enhance(self, image):
        """
        Thực hiện toàn bộ pipeline khôi phục:
        1. Khử bóng (Illumination Normalization)
        2. Tăng cường chi tiết và màu nền giấy
        3. Tạo mask cho chữ viết nội dung
        """
        # 1. Khử bóng đổ
        normalized_img = self.remove_shadows_division(image)
        
        # 2. Xử lý nền trắng và Mask chữ
        final_img = self.smart_binarize(normalized_img)
        
        return final_img

    def remove_shadows_division(self, image):
        """
        Phương pháp "Division-based Illumination Normalization".
        Dùng để khử cực sạch các bóng đen/ngón tay ám trên giấy.
        """
        # Chuyển xám để lấy bản đồ ánh sáng
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Tạo Background Map (Ảnh nền chỉ chứa bóng râm, mất hết chữ)
        # Sử dụng Dilation kết hợp Blur cực mạnh
        kernel_size = 21 # Tuỳ chỉnh theo độ phân giải ảnh
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Đóng hình thái học (Lấp đầy các nét chữ đen bằng màu nền trắng xung quanh)
        bg_estimate = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        bg_estimate = cv2.GaussianBlur(bg_estimate, (51, 51), 0)
        
        # Chia ảnh gốc cho ảnh nền: I_out = 255 * (I / Bg)
        # Cộng thêm epsilon để tránh chia cho 0
        normalized_float = 255.0 * (gray.astype(float) / (bg_estimate.astype(float) + 1e-6))
        
        # Đưa về lại dải 0-255
        normalized_gray = np.clip(normalized_float, 0, 255).astype(np.uint8)
        
        return normalized_gray

    def smart_binarize(self, gray_image):
        """
        Thay vì Binarize cứng toàn bộ ảnh thành Trắng/Đen làm gãy nét chữ,
        ta chỉ lấy Threhold làm Mask, vùng nào là nền sẽ ép Trắng Tinh.
        Vùng nét chữ giữ lại độ chống răng cưa (Anti-aliasing) bằng cách 
        tăng contrast/sharpening ảnh gốc.
        """
        # Tăng cường độ gắt của chữ bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray_image)
        
        # Tạo mask nhị phân tìm vị trí nét chữ
        # blockSize lớn để phân vùng rộng, C lớn để lọc nhiễu nhẹ
        text_mask = cv2.adaptiveThreshold(
            enhanced_gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            31, 15
        )
        
        # Dọn dẹp mask (loại bỏ hạt muối nhỏ)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel)
        
        # Tạo Output: Mặc định là giấy trắng tinh
        final_img = np.full_like(enhanced_gray, 255)
        
        # Vùng nào là CHỮ (trong clean_mask giá trị là 0 vì cv2 threshold trả mask đảo ngược -> đen=chữ, trắng=nền)
        # Copy pixel cường độ thật từ enhanced_gray sang final_img
        # Chữ đen có giá trị < 255 -> text_mask == 0
        text_pixels = (clean_mask == 0)
        final_img[text_pixels] = enhanced_gray[text_pixels]
        
        return final_img

if __name__ == "__main__":
    print("Test Enhancement...")
