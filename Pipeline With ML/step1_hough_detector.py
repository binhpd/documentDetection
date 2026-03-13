"""
Step 1d: Fallback — Hough Line Transform + Line Intersection
🟢 Image Processing

Chức năng:
- Khi approxPolyDP thất bại (góc bị che, viền đứt đoạn)
- Dùng HoughLinesP tìm tất cả đoạn thẳng
- Gom cụm thành 4 nhóm (Trên / Dưới / Trái / Phải)
- Tính giao điểm toán học → 4 góc ảo

Input:  Ảnh nhị phân cạnh (H × W, 0/255)
Output: Mảng 4 toạ độ giao điểm (4, 2) float32 — hoặc None

Thuật toán chính:
1. HoughLinesP → danh sách đoạn thẳng (x1,y1,x2,y2)
2. Phân loại mỗi đoạn: Ngang (angle < 45°) hoặc Dọc (angle >= 45°)
3. Ngang → chia thành Trên/Dưới theo toạ độ y trung bình
4. Dọc → chia thành Trái/Phải theo toạ độ x trung bình
5. Fit đường thẳng đại diện cho mỗi nhóm (Linear Regression)
6. Tính 4 giao điểm: TL = Trên ∩ Trái, TR = Trên ∩ Phải, ...
"""

import cv2
import numpy as np


class HoughCornerDetector:
    def __init__(self, rho=1, theta_res=np.pi/180, threshold=50,
                 min_line_length=50, max_line_gap=10):
        """
        Args:
            rho: Độ phân giải khoảng cách (pixel) trong Hough space
            theta_res: Độ phân giải góc (radian) trong Hough space
            threshold: Số vote tối thiểu để coi là đường thẳng
            min_line_length: Chiều dài tối thiểu của đoạn thẳng (pixel)
            max_line_gap: Khoảng trống tối đa được phép trong 1 đoạn (pixel)
        """
        self.rho = rho
        self.theta_res = theta_res
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def _get_lines(self, edged_image):
        """Tìm tất cả đoạn thẳng bằng HoughLinesP."""
        lines = cv2.HoughLinesP(
            edged_image,
            rho=self.rho,
            theta=self.theta_res,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        if lines is None:
            return []
        return lines.reshape(-1, 4)  # (N, 4) → mỗi dòng là (x1, y1, x2, y2)

    def _classify_lines(self, lines, img_height, img_width):
        """Phân loại đoạn thẳng thành 4 nhóm: Trên, Dưới, Trái, Phải.
        
        Bước 1: Tính góc nghiêng → Ngang (< 45°) hoặc Dọc (>= 45°)
        Bước 2: Ngang → chia theo y trung bình (< giữa = Trên, >= giữa = Dưới)
        Bước 3: Dọc → chia theo x trung bình (< giữa = Trái, >= giữa = Phải)
        """
        horizontal = []  # Các đoạn ngang
        vertical = []    # Các đoạn dọc

        for x1, y1, x2, y2 in lines:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 45 or angle > 135:
                horizontal.append((x1, y1, x2, y2))
            else:
                vertical.append((x1, y1, x2, y2))

        if len(horizontal) < 2 or len(vertical) < 2:
            return None

        # Chia ngang thành Trên/Dưới
        mid_y = img_height / 2
        top_lines = [(x1, y1, x2, y2) for x1, y1, x2, y2 in horizontal
                     if (y1 + y2) / 2 < mid_y]
        bottom_lines = [(x1, y1, x2, y2) for x1, y1, x2, y2 in horizontal
                        if (y1 + y2) / 2 >= mid_y]

        # Chia dọc thành Trái/Phải
        mid_x = img_width / 2
        left_lines = [(x1, y1, x2, y2) for x1, y1, x2, y2 in vertical
                      if (x1 + x2) / 2 < mid_x]
        right_lines = [(x1, y1, x2, y2) for x1, y1, x2, y2 in vertical
                       if (x1 + x2) / 2 >= mid_x]

        if not all([top_lines, bottom_lines, left_lines, right_lines]):
            return None

        return {
            'top': top_lines,
            'bottom': bottom_lines,
            'left': left_lines,
            'right': right_lines
        }

    def _fit_line(self, line_segments):
        """Fit 1 đường thẳng đại diện cho nhóm đoạn thẳng.
        
        Gom tất cả điểm đầu-cuối của các đoạn, dùng cv2.fitLine
        để tìm đường thẳng tối ưu (least squares).
        
        Returns:
            (vx, vy, x0, y0): Vector hướng + 1 điểm trên đường thẳng
        """
        points = []
        for x1, y1, x2, y2 in line_segments:
            points.append([x1, y1])
            points.append([x2, y2])
        points = np.array(points, dtype=np.float32)

        # cv2.fitLine trả về [vx, vy, x0, y0]
        line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line.flatten()
        return (vx, vy, x0, y0)

    def _intersect(self, line1, line2):
        """Tính giao điểm của 2 đường thẳng.
        
        Đường thẳng được biểu diễn dạng tham số:
            P = P0 + t * V
        
        Giải hệ phương trình:
            x0_1 + t1 * vx1 = x0_2 + t2 * vx2
            y0_1 + t1 * vy1 = y0_2 + t2 * vy2
        
        Returns:
            (x, y): Toạ độ giao điểm — hoặc None nếu song song
        """
        vx1, vy1, x01, y01 = line1
        vx2, vy2, x02, y02 = line2

        # Giải bằng định thức (Cramer's rule)
        det = vx1 * vy2 - vy1 * vx2
        if abs(det) < 1e-10:
            return None  # 2 đường song song

        dx = x02 - x01
        dy = y02 - y01
        t1 = (dx * vy2 - dy * vx2) / det

        x = x01 + t1 * vx1
        y = y01 + t1 * vy1
        return (x, y)

    def find_corners(self, edged_image):
        """Tìm 4 góc tài liệu bằng Hough Lines + Line Intersection.
        
        Returns:
            corners: np.array shape (4, 2) float32 — hoặc None
        """
        h, w = edged_image.shape[:2]

        # Bước 1: Tìm tất cả đoạn thẳng
        lines = self._get_lines(edged_image)
        if len(lines) < 4:
            return None

        # Bước 2: Phân loại thành 4 nhóm
        groups = self._classify_lines(lines, h, w)
        if groups is None:
            return None

        # Bước 3: Fit đường thẳng đại diện cho mỗi nhóm
        top_line = self._fit_line(groups['top'])
        bottom_line = self._fit_line(groups['bottom'])
        left_line = self._fit_line(groups['left'])
        right_line = self._fit_line(groups['right'])

        # Bước 4: Tính 4 giao điểm
        tl = self._intersect(top_line, left_line)       # Top-Left
        tr = self._intersect(top_line, right_line)      # Top-Right
        br = self._intersect(bottom_line, right_line)   # Bottom-Right
        bl = self._intersect(bottom_line, left_line)    # Bottom-Left

        if any(p is None for p in [tl, tr, br, bl]):
            return None

        corners = np.array([tl, tr, br, bl], dtype=np.float32)

        # Kiểm tra toạ độ hợp lệ (trong phạm vi ảnh, cho phép margin 10%)
        margin = max(h, w) * 0.1
        for x, y in corners:
            if x < -margin or x > w + margin or y < -margin or y > h + margin:
                return None

        return corners
