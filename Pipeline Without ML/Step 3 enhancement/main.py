import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Step 1 canny edge detection"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Step 2 perspective transform"))

from preprocessor import ImagePreprocessor
from edge_detector import EdgeDetector
from corner_detector import CornerDetector
from perspective_transform import PerspectiveTransform
from shadow_removal import ShadowRemoval
from adaptive_threshold import AdaptiveThreshold
from morphology import MorphologyCleaner


def show_pipeline_results(rows, main_title="Pipeline Results"):
    """Hiển thị kết quả pipeline theo nhiều hàng.

    Args:
        rows: list of (row_title, [(col_title, image), ...])
    """
    n_rows = len(rows)
    n_cols = max(len(cols) for _, cols in rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    for row_axes in axes:
        if not hasattr(row_axes, "__iter__"):
            row_axes = [row_axes]

    fig.suptitle(main_title, fontsize=18, fontweight="bold", y=1.02)

    for r, (row_title, cols) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c] if hasattr(axes[r], "__iter__") else axes[r]
            if c < len(cols):
                title, img = cols[c]
                if len(img.shape) == 2:
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title(title, fontsize=11, fontweight="bold")
            ax.axis("off")

        if hasattr(axes[r], "__iter__"):
            axes[r][0].set_ylabel(row_title, fontsize=13, fontweight="bold",
                                  rotation=0, labelpad=80, va="center")

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) >= 3:
        category = sys.argv[1]
        img_idx = sys.argv[2]
        sample_image = os.path.join(os.path.dirname(__file__), "..", "..", "image", category, f"{int(img_idx):04d}.jpg")
    elif len(sys.argv) == 2:
        sample_image = sys.argv[1]
    else:
        sample_image = os.path.join(os.path.dirname(__file__), "..", "Step 1 canny edge detection", "sample_receipt.jpg")
        
    print(f"File ảnh đầu vào: {sample_image}")

    if not os.path.exists(sample_image):
        print(f"Không tìm thấy ảnh mẫu: {sample_image}")
        return

    # ═══════════════════════════════════════════════════════════════
    #  STEP 1: Phát hiện vùng tài liệu (Edge Detection + Corners)
    # ═══════════════════════════════════════════════════════════════
    img = cv2.imread(sample_image)
    ratio = img.shape[0] / 500.0
    orig = img.copy()
    img_resized = cv2.resize(img, (int(img.shape[1] / ratio), 500))

    preprocessor = ImagePreprocessor()
    processed = preprocessor.process(img_resized)
    print("[Step 1a] Tiền xử lý: Grayscale + Gaussian Blur   ✓")

    edge_detector = EdgeDetector()
    edged = edge_detector.detect(processed)
    print("[Step 1b] Phát hiện cạnh: Canny Edge Detection    ✓")

    corner_detector = CornerDetector()
    corners = corner_detector.find_corners(edged)

    if corners is None:
        print("[Step 1c] Không tìm thấy 4 góc tài liệu          ✗")
        return

    print("[Step 1c] Tìm 4 góc tài liệu                     ✓")

    corners_vis = orig.copy()
    cv2.drawContours(corners_vis,
                     [np.int32(corners.reshape(4, 2) * ratio)], -1, (0, 255, 0), 3)

    # ═══════════════════════════════════════════════════════════════
    #  STEP 2: Perspective Transform
    # ═══════════════════════════════════════════════════════════════
    pts_on_orig = corners.reshape(4, 2).astype(np.float32) * ratio
    transformer = PerspectiveTransform()
    warped, sorted_pts = transformer.transform(orig, pts_on_orig)
    print(f"[Step 2]  Biến đổi phối cảnh (Perspective Warp)  ✓")
    print(f"          Kích thước: {warped.shape[1]}×{warped.shape[0]} px")

    # ═══════════════════════════════════════════════════════════════
    #  STEP 3: Xử lý Ánh sáng & Tương phản
    # ═══════════════════════════════════════════════════════════════
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # -- 3a: Loại bỏ bóng đổ --
    shadow_remover = ShadowRemoval(kernel_size=7, morph_iterations=2)
    shadow_free, bg_estimate = shadow_remover.remove(warped_gray)
    print("[Step 3a] Loại bỏ bóng đổ (Shadow Removal)       ✓")

    # -- 3b: Adaptive Thresholding (Gaussian) --
    thresh_gaussian = AdaptiveThreshold(block_size=11, C=7, method="gaussian")
    binary_gaussian = thresh_gaussian.apply(shadow_free)
    print("[Step 3b] Adaptive Threshold (Gaussian)           ✓")

    # -- 3b': Adaptive Thresholding (Mean) — so sánh --
    thresh_mean = AdaptiveThreshold(block_size=11, C=7, method="mean")
    binary_mean = thresh_mean.apply(shadow_free)
    print("[Step 3b'] Adaptive Threshold (Mean) — so sánh    ✓")

    # -- 3c: Morphological Cleaning --
    cleaner = MorphologyCleaner(open_kernel_size=2, close_kernel_size=2)
    cleaned_gaussian = cleaner.clean(binary_gaussian)
    cleaned_mean = cleaner.clean(binary_mean)
    print("[Step 3c] Morphological Cleaning (Open + Close)   ✓")

    # ═══════════════════════════════════════════════════════════════
    #  HIỂN THỊ KẾT QUẢ
    # ═══════════════════════════════════════════════════════════════
    print("\n══════════════════════════════════════════════════")
    print("  Hiển thị kết quả toàn bộ pipeline...")
    print("══════════════════════════════════════════════════")

    show_pipeline_results([
        ("Step 1-2", [
            ("Ảnh gốc", orig),
            ("1b. Canny Edges", edged),
            ("1c. 4 góc tài liệu", corners_vis),
            ("2. Perspective Warp", warped),
        ]),
        ("Step 3", [
            ("3a. Bỏ bóng đổ", shadow_free),
            ("3b. Adaptive (Gaussian)", binary_gaussian),
            ("3b'. Adaptive (Mean)", binary_mean),
            ("3c. Morphology Clean", cleaned_gaussian),
        ]),
    ], main_title="Document Scanner Pipeline — Step 1 → 2 → 3")


if __name__ == "__main__":
    main()
