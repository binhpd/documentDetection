"""
Tiện ích: Sắp xếp 4 góc theo thứ tự [TL, TR, BR, BL]
🟢 Image Processing

Thuật toán:
- Top-Left:     tổng (x+y) nhỏ nhất
- Top-Right:    hiệu (y-x) nhỏ nhất  (x lớn, y nhỏ)
- Bottom-Right: tổng (x+y) lớn nhất
- Bottom-Left:  hiệu (y-x) lớn nhất  (x nhỏ, y lớn)
"""

import numpy as np


class CornerSorter:
    @staticmethod
    def sort(pts):
        """Sắp xếp 4 điểm thành [TL, TR, BR, BL].
        
        Args:
            pts: numpy array shape (4, 2) hoặc (4, 1, 2)
        Returns:
            numpy array shape (4, 2) dtype float32
        """
        pts = pts.reshape(4, 2).astype(np.float32)
        sorted_pts = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)       # x + y
        d = np.diff(pts, axis=1)  # y - x

        sorted_pts[0] = pts[np.argmin(s)]   # Top-Left
        sorted_pts[1] = pts[np.argmin(d)]   # Top-Right
        sorted_pts[2] = pts[np.argmax(s)]   # Bottom-Right
        sorted_pts[3] = pts[np.argmax(d)]   # Bottom-Left

        return sorted_pts
