import cv2
import numpy as np
from corner_sorter import CornerSorter


class PerspectiveTransform:
    """Biến đổi phối cảnh: ánh xạ vùng tài liệu (tứ giác) về hình chữ nhật phẳng.

    Pipeline:
      1. Sắp xếp 4 đỉnh → [TL, TR, BR, BL]
      2. Tính kích thước ảnh đích (W × H) từ khoảng cách Euclidean
      3. Tính ma trận biến đổi M (3×3) bằng getPerspectiveTransform
      4. Áp dụng warpPerspective
    """

    def __init__(self):
        self.sorter = CornerSorter()

    def transform(self, image, pts):
        """Thực hiện perspective warp.

        Args:
            image: ảnh gốc (BGR, chưa resize)
            pts: 4 đỉnh tài liệu trên ảnh gốc, shape (4,1,2) hoặc (4,2)

        Returns:
            warped: ảnh đã duỗi phẳng
            sorted_pts: 4 đỉnh đã sắp xếp (để vẽ debug)
        """
        sorted_pts = self.sorter.sort(pts)
        tl, tr, br, bl = sorted_pts

        width = self._compute_width(tl, tr, bl, br)
        height = self._compute_height(tl, tr, bl, br)

        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(sorted_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (int(width), int(height)))

        return warped, sorted_pts

    @staticmethod
    def _compute_width(tl, tr, bl, br):
        """W = max(|TR - TL|, |BR - BL|)"""
        w1 = np.linalg.norm(tr - tl)
        w2 = np.linalg.norm(br - bl)
        return max(w1, w2)

    @staticmethod
    def _compute_height(tl, tr, bl, br):
        """H = max(|TL - BL|, |TR - BR|)"""
        h1 = np.linalg.norm(tl - bl)
        h2 = np.linalg.norm(tr - br)
        return max(h1, h2)
