import cv2
import numpy as np

class CornerDetector:
    def __init__(self, approx_epsilon=0.02, use_hough_fallback=True):
        self.approx_epsilon = approx_epsilon
        self.use_hough_fallback = use_hough_fallback

    def get_intersection(self, line1, line2):
        """Tìm giao điểm của 2 đường thẳng (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None # Các đường song song

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return (int(px), int(py))

    def _get_distance(self, pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def _cluster_lines(self, lines, max_dist=50, max_angle=10):
        """Gom cụm các đường thẳng gần nhau (tránh trả về nhiều đường thẳng trùng lấp)."""
        # (Để đơn giản, trong phiên bản fallback này ta sẽ giữ các dòng dài nhất hoặc sử dụng RANSAC/KMeans)
        # Tạm thời chỉ sắp xếp theo độ dài giảm dần
        if lines is None: return []
        lines_info = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = self._get_distance((x1,y1), (x2,y2))
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            lines_info.append((length, angle, line[0]))
        
        lines_info.sort(key=lambda x: x[0], reverse=True)
        return [info[2] for info in lines_info]

    def _fallback_hough_lines(self, edged_image):
        """Sử dụng Hough Lines để tìm 4 cạnh lớn nhất và tìm giao điểm."""
        # Lấy các đường thẳng từ ảnh cạnh
        lines = cv2.HoughLinesP(edged_image, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)
        if lines is None: return None

        sorted_lines = self._cluster_lines(lines)
        if len(sorted_lines) < 4: return None

        # Chia đường thẳng thành 2 nhóm: Ngang và Dọc
        horizontal_lines = []
        vertical_lines = []
        for line in sorted_lines:
            x1, y1, x2, y2 = line
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            if angle < 45 or angle > 135:
                # Gần ngang
                horizontal_lines.append(line)
            else:
                vertical_lines.append(line)

        # Cần ít nhất 2 đường ngang và 2 đường dọc để tạo 4 góc
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2: return None

        # Lấy 2 đường ngang dài nhất (tương đối) và cách xa nhau nhất
        # Tạm thời lấy 2 đường dài nhất nằm ở phía trên và phía dưới
        h_y_coords = [(line, (line[1] + line[3])/2) for line in horizontal_lines]
        h_y_coords.sort(key=lambda x: x[1]) # từ trên xuống dưới
        top_line = h_y_coords[0][0]
        bottom_line = h_y_coords[-1][0]

        v_x_coords = [(line, (line[0] + line[2])/2) for line in vertical_lines]
        v_x_coords.sort(key=lambda x: x[1]) # từ trái qua phải
        left_line = v_x_coords[0][0]
        right_line = v_x_coords[-1][0]

        # Tìm 4 giao điểm
        tl = self.get_intersection(top_line, left_line)
        tr = self.get_intersection(top_line, right_line)
        bl = self.get_intersection(bottom_line, left_line)
        br = self.get_intersection(bottom_line, right_line)

        if tl and tr and bl and br:
            corners = np.array([[tl], [tr], [br], [bl]], dtype=np.int32)
            # Kiểm tra xem các góc có lọt trong khung hình hay xấp xỉ không?
            # Đơn giản thì ở đây cứ trả về
            return corners
            
        return None

    def find_corners(self, edged_image):
        """Tìm 4 góc của tài liệu từ ảnh đã phát hiện biên"""
        contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Sắp xếp contours theo diện tích giảm dần, lấy 5 contour lớn nhất
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in contours:
            peri = cv2.arcLength(c, True)
            # Xấp xỉ đa giác
            approx = cv2.approxPolyDP(c, self.approx_epsilon * peri, True)

            # Nếu khung xấp xỉ có đúng 4 đỉnh
            if len(approx) == 4:
                return approx
        
        # Nếu không tìm thấy 4 đỉnh bằng contour thông thường (do bị che khuất / đứt đoạn)
        if self.use_hough_fallback:
            print("  -> (Fallback) approxPolyDP thất bại, đang thử Hough Lines Intersection...")
            fallback_corners = self._fallback_hough_lines(edged_image)
            if fallback_corners is not None:
                return fallback_corners
                
        return None
