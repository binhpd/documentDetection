"""
Step 2: Biến đổi phối cảnh (Perspective Transform)
🟢 Image Processing

Chức năng:
- Nhận input là ảnh gốc và 4 toạ độ góc (từ Step 1)
- Áp dụng 'Bird's-eye view' (nhìn từ trên xuống) để bẻ phẳng và làm vuông vức mặt phẳng tài liệu.
- Tự động tính toán kích thước chiều rộng/chiều cao đầu ra dựa trên khoảng cách giữa các góc.

Input:  
- image: Ảnh BGR gốc (H × W × 3, uint8)
- corners: np.array (4, 2) float32 chứa toạ độ 4 góc [Top-Left, Top-Right, Bottom-Right, Bottom-Left]

Output: 
- warped: Ảnh BGR chỉ chứa bề mặt tài liệu đã được bẻ thẳng, cắt bỏ toàn bộ nền ngoài.
"""

import cv2
import numpy as np

class PerspectiveTransformer:
    def __init__(self):
        pass

    def _order_points(self, pts):
        """
        Sắp xếp 4 điểm theo thứ tự:
        Top-Left, Top-Right, Bottom-Right, Bottom-Left
        (Dù Step 1 đã sắp xếp nhưng bước này đảm bảo tính toàn vẹn độc lập của Step 2)
        """
        rect = np.zeros((4, 2), dtype="float32")

        # Top-left có tổng nhỏ nhất, Bottom-right có tổng lớn nhất
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-right có hiệu (y-x) nhỏ nhất, Bottom-left có hiệu lớn nhất
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def transform(self, image, corners):
        """
        Thực hiện biến đổi phối cảnh 4 góc (Perspective Transform).
        
        Args:
            image: Ảnh gốc (numpy array).
            corners: Mảng 4 điểm (4, 2) toạ độ pixel trên ảnh gốc.
            
        Returns:
            warped: Ảnh mặt tài liệu đã được bẻ thẳng và cắt (crop).
        """
        if corners is None or len(corners) != 4:
            print("❌ [Step 2] Không đủ 4 điểm góc. Bỏ qua Perspective Transform.")
            return image

        # Đảm bảo thứ tự chuẩn
        rect = self._order_points(corners)
        (tl, tr, br, bl) = rect

        # --- TÍNH TOÁN KÍCH THƯỚC ẢNH ĐẦU RA (Destination Image) ---
        # 1. Chiều rộng (Width) mới: Lớn nhất giữa cạnh trên (tr-tl) và cạnh dưới (br-bl)
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # 2. Chiều cao (Height) mới: Lớn nhất giữa cạnh trái (bl-tl) và cạnh phải (br-tr)
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Bảo vệ chống chia cho 0/kích thước lỗi
        if maxWidth == 0 or maxHeight == 0:
             print("❌ [Step 2] Kích thước toán học lỗi (width hoặc height = 0).")
             return image

        # --- ĐỊNH NGHĨA 4 ĐIỂM ĐÍCH (Destination Points) ---
        # Dóng 4 góc vào một hình chữ nhật hoàn hảo từ toạ độ (0,0) tới (maxWidth-1, maxHeight-1)
        # Thứ tự tương ứng với `rect`: TL, TR, BR, BL
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # --- TÍNH MA TRẬN PHỐI CẢNH (Perspective Matrix) ---
        # Tìm ma trận biến đổi M kích thước 3x3 giúp map từ rect sang dst
        M = cv2.getPerspectiveTransform(rect, dst)

        # --- ÁP DỤNG BIẾN ĐỔI ---
        # "Dán" ảnh gốc lên hình chữ nhật chuẩn qua phép biến đổi M
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        # Bớt một chút lề nhỏ ở mép (tùy chọn) để gọt bỏ rìa đen hoặc nền dính vào
        # Chạy hiệu quả nếu dùng OpenCV thuần túy
        crop_margin = 0 
        if crop_margin > 0:
            warped = warped[crop_margin:warped.shape[0]-crop_margin, crop_margin:warped.shape[1]-crop_margin]

        return warped

# Chỉ dùng cho testing độc lập
if __name__ == "__main__":
    test_img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.polylines(test_img, [np.array([[100, 100], [400, 50], [450, 450], [50, 400]])], True, (0, 255, 0), 2)
    transformer = PerspectiveTransformer()
    # Mock corners
    mock_corners = np.array([[100, 100], [400, 50], [450, 450], [50, 400]], dtype="float32")
    res = transformer.transform(test_img, mock_corners)
    print(f"Test transform: shape {res.shape}")
