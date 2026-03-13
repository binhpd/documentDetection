import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Step 1 canny edge detection"))

from preprocessor import ImagePreprocessor
from edge_detector import EdgeDetector
from corner_detector import CornerDetector
from perspective_transform import PerspectiveTransform


def draw_sorted_corners(image, sorted_pts, ratio):
    """Vẽ 4 đỉnh đã sắp xếp lên ảnh với nhãn TL/TR/BR/BL."""
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    annotated = image.copy()

    pts_orig = (sorted_pts * ratio).astype(int)
    cv2.drawContours(annotated, [pts_orig], -1, (0, 255, 0), 3)

    for i, (pt, label, color) in enumerate(zip(pts_orig, labels, colors)):
        cv2.circle(annotated, tuple(pt), 12, color, -1)
        cv2.putText(annotated, label, (pt[0] + 15, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    return annotated


def show_pipeline_results(steps):
    """Hiển thị kết quả từng bước trên cùng 1 cửa sổ bằng matplotlib."""
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, steps):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    step1_dir = os.path.join(os.path.dirname(__file__), "..", "Step 1 canny edge detection")
    sample_image = "/Users/binhpham/Documents/Study/MSE/Xử lý ảnh Video/Bài tập cuối kỳ/Nhóm 6/image/rotate/0067.jpg"

    if not os.path.exists(sample_image):
        print(f"Không tìm thấy ảnh mẫu: {sample_image}")
        print("Đặt ảnh 'sample_receipt.jpg' vào thư mục 'Step 1 canny edge detection/'")
        return

    # ── Đọc ảnh gốc ──
    img = cv2.imread(sample_image)
    ratio = img.shape[0] / 500.0
    orig = img.copy()
    img_resized = cv2.resize(img, (int(img.shape[1] / ratio), 500))

    # ── STEP 1a: Tiền xử lý (Grayscale + GaussianBlur) ──
    preprocessor = ImagePreprocessor()
    processed = preprocessor.process(img_resized)
    print("[Step 1a] Tiền xử lý: Grayscale + Gaussian Blur  ✓")

    # ── STEP 1b: Canny Edge Detection ──
    edge_detector = EdgeDetector()
    edged = edge_detector.detect(processed)
    print("[Step 1b] Phát hiện cạnh: Canny Edge Detection   ✓")

    # ── STEP 1c: Tìm 4 góc tài liệu ──
    corner_detector = CornerDetector()
    corners = corner_detector.find_corners(edged)

    if corners is None:
        print("[Step 1c] Không tìm thấy 4 góc tài liệu ✗")
        show_pipeline_results([
            ("Ảnh gốc", orig),
            ("1a. Grayscale + Blur", processed),
            ("1b. Canny Edges", edged),
        ])
        return

    print(f"[Step 1c] Tìm 4 góc tài liệu                   ✓")

    corners_on_orig = draw_sorted_corners(img, corners.reshape(4, 2).astype(np.float32), ratio)

    # ── STEP 2: Perspective Transform ──
    pts_on_orig = corners.reshape(4, 2).astype(np.float32) * ratio
    transformer = PerspectiveTransform()
    warped, sorted_pts = transformer.transform(orig, pts_on_orig)
    print(f"[Step 2]  Biến đổi phối cảnh (Perspective Warp) ✓")
    print(f"          Kích thước ảnh đầu ra: {warped.shape[1]}×{warped.shape[0]} px")

    # ── Hiển thị tất cả các bước ──
    show_pipeline_results([
        ("Ảnh gốc", orig),
        ("1a. Grayscale + Blur", processed),
        ("1b. Canny Edges", edged),
        ("1c. Phát hiện 4 góc", corners_on_orig),
        ("2. Perspective Warp", warped),
    ])


if __name__ == "__main__":
    main()
