import numpy as np


class CornerSorter:
    """Sắp xếp 4 đỉnh tài liệu theo thứ tự: top-left, top-right, bottom-right, bottom-left.

    Thuật toán:
      - top-left: tổng (x+y) nhỏ nhất
      - bottom-right: tổng (x+y) lớn nhất
      - top-right: hiệu (y-x) nhỏ nhất  (x lớn, y nhỏ)
      - bottom-left: hiệu (y-x) lớn nhất (x nhỏ, y lớn)
    """

    @staticmethod
    def sort(pts):
        """Nhận mảng 4 điểm bất kỳ, trả về mảng đã sắp xếp [TL, TR, BR, BL].

        Args:
            pts: numpy array shape (4, 1, 2) hoặc (4, 2)

        Returns:
            numpy array shape (4, 2) dtype float32
        """
        pts = pts.reshape(4, 2).astype(np.float32)

        sorted_pts = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)        # x + y
        d = np.diff(pts, axis=1)   # y - x

        sorted_pts[0] = pts[np.argmin(s)]   # top-left
        sorted_pts[1] = pts[np.argmin(d)]   # top-right
        sorted_pts[2] = pts[np.argmax(s)]   # bottom-right
        sorted_pts[3] = pts[np.argmax(d)]   # bottom-left

        return sorted_pts
