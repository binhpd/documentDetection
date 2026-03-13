"""
STEP 1 — Main Orchestrator
Điều phối toàn bộ Step 1: Phát hiện vùng tài liệu

Chiến lược Cascading Fallback:
    1c. approxPolyDP     🟢 Image Processing  ← Thử trước
         ↓ thất bại
    1d. Hough Lines      🟢 Image Processing  ← Fallback 1
         ↓ thất bại
    1e. ML Segmentation  🔴 Machine Learning  ← Fallback cuối (tuỳ chọn)

Usage:
    python main.py <đường_dẫn_ảnh>
    python main.py  (sử dụng ảnh mẫu mặc định)
"""

import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from step1_preprocessor import Preprocessor
from step1_edge_detector import EdgeDetector
from step1_hed_detector import HEDDetector
from step1_contour_detector import ContourCornerDetector
from step1_hough_detector import HoughCornerDetector
from step1_ml_segmentor import MLSegmentor
from step1_docaligner import DocAlignerSegmentor
from step2_perspective_transform import PerspectiveTransformer
from step2_ml_dewarper import MLDewarper
from step3_enhancer import DocumentEnhancer
from corner_sorter import CornerSorter


class DocumentDetector:
    """Điều phối Step 1 với chiến lược cascading fallback."""

    def __init__(self, enable_ml=False, use_hed=False, use_docaligner=False, yolo_model="models/yolov8n-seg.pt", unet_model="models/unet_doc.pt", use_ml_dewarp=False):
        """
        Khởi tạo DocumentDetector pipeline.
        
        Args:
            enable_ml: Có bật ML fallback không (YOLOv8)
            use_hed: Có dùng HED Edge Detection thay cho Canny không
            use_docaligner: Sử dụng DocAligner chuyên dụng làm segmentor
            use_ml_dewarp: Sử dụng ML Dewarping phi tuyến tính thay thế Perspective Transform cho tài liệu cong
        """
        self.preprocessor = Preprocessor()
        self.use_hed = use_hed
        if use_hed:
            model_dir = os.path.join(os.path.dirname(__file__), "models")
            self.hed_detector = HEDDetector(
                os.path.join(model_dir, "deploy.prototxt"),
                os.path.join(model_dir, "hed_pretrained_bsds.caffemodel")
            )
        else:
            self.edge_detector = EdgeDetector()

        # --- Models ---
        self.contour_detector = ContourCornerDetector()
        self.hough_detector = HoughCornerDetector()
        
        # --- Step 2: Transformer ---
        baseline_transformer = PerspectiveTransformer()
        if use_ml_dewarp:
            self.transformer = MLDewarper(fallback_transformer=baseline_transformer)
        else:
            self.transformer = baseline_transformer
            
        self.enable_ml = enable_ml
        self.use_docaligner = use_docaligner
        
        self.ml_segmentor = MLSegmentor(yolo_model, unet_model) if enable_ml else None
        self.docaligner_segmentor = DocAlignerSegmentor() if use_docaligner else None

        # --- Step 3: Enhancer ---
        self.enhancer = DocumentEnhancer()

    def detect(self, image):
        """Phát hiện 4 góc tài liệu trong ảnh.
        
        Args:
            image: Ảnh BGR gốc (H × W × 3, uint8)
            
        Returns:
            result: dict chứa:
                - 'corners': np.array (4, 2) float32 trên ảnh gốc — hoặc None
                - 'method': Tên phương pháp đã dùng ('contour' / 'hough' / 'ml' / None)
                - 'edged': Ảnh cạnh Canny
                - 'blurred': Ảnh xám đã blur
                - 'resized': Ảnh màu đã resize
                - 'ratio': Tỷ lệ scale
                - 'mask': Mask ML (nếu dùng ML, None nếu không)
        """
        orig = image.copy()

        # ── 1a. Tiền xử lý ──
        blurred, resized, ratio = self.preprocessor.process(image)
        print(f"[1a] Tiền xử lý: {image.shape} → resize {resized.shape}, ratio={ratio:.2f}  ✓")

        # ── 1b. Edge Detection (HED hoặc Canny) ──
        if self.use_hed:
            edged = self.hed_detector.detect(resized)
            if edged is None:
                print("❌ [HED] Fallback về Canny vì HED lỗi.")
                # Tạm thời tạo Canny
                edged = EdgeDetector().detect(blurred)
                print(f"[1b] Canny Edge Detection (fallback)  ✓")
            else:
                print(f"[1b] HED Edge Detection  ✓")
        else:
            edged = self.edge_detector.detect(blurred)
            print(f"[1b] Canny Edge Detection (auto-threshold)  ✓")

        result = {
            'corners': None,
            'method': None,
            'edged': edged,
            'blurred': blurred,
            'resized': resized,
            'ratio': ratio,
            'mask': None,
        }

        # ── 1c. approxPolyDP (Phương pháp chính) ──
        corners = self.contour_detector.find_corners(edged)
        if corners is not None:
            print(f"[1c] approxPolyDP → Tìm được 4 góc  ✓")
            result['corners'] = corners * ratio  # Quy về ảnh gốc
            result['method'] = 'contour'
            return result
        else:
            print(f"[1c] approxPolyDP → Không tìm được 4 góc  ✗")

        # ── 1d. Hough Lines Fallback ──
        corners = self.hough_detector.find_corners(edged)
        if corners is not None:
            print(f"[1d] Hough Lines + Intersection → Tìm được 4 góc  ✓")
            result['corners'] = corners * ratio
            result['method'] = 'hough'
            return result
        else:
            print(f"[1d] Hough Lines → Không tìm được 4 góc  ✗")

        # ── 1e. ML Segmentation (DocAligner hoặc YOLO) ──
        if self.use_docaligner and self.docaligner_segmentor is not None:
            print(f"[1e] Thử DocAligner Segmentation...")
            # DocAligner có thể chạy trực tiếp trên ảnh gốc do module của nó tự handle scale tốt hơn
            # Tuy nhiên để nhất quán với Preprocessor, ta cứ pass resized image:
            mask, corners = self.docaligner_segmentor.segment(resized)
            if corners is not None:
                print(f"[1e] DocAligner → Tìm được 4 góc cực khít  ✓")
                result['corners'] = corners * ratio
                result['method'] = 'docaligner'
                result['mask'] = mask
                return result
            else:
                print(f"[1e] DocAligner → Thất bại  ✗. Sẽ fallback về YOLO.")

        if self.enable_ml and self.ml_segmentor is not None:
            print(f"[1e] Thử YOLO ML Segmentation...")
            mask, corners = self.ml_segmentor.segment(resized)
            if corners is not None:
                print(f"[1e] YOLO ML Segmentation → Tìm được 4 góc  ✓")
                result['corners'] = corners * ratio
                result['method'] = 'ml'
                result['mask'] = mask
                return result
            else:
                print(f"[1e] ML Segmentation → Thất bại  ✗")
        else:
            print(f"[1e] ML Segmentation → Bỏ qua (chưa bật/chưa cài)")

        print(f"\n❌ THẤT BẠI: Không thể tìm 4 góc tài liệu bằng bất kỳ phương pháp nào.")
        return result


def draw_corners(image, corners, method, ratio=1.0):
    """Vẽ 4 góc lên ảnh để hiển thị."""
    vis = image.copy()
    pts = corners.astype(int)
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]

    # Sắp xếp góc
    sorted_pts = CornerSorter.sort(pts)

    # Vẽ đường viền
    cv2.drawContours(vis, [sorted_pts.astype(int)], -1, (0, 255, 0), 3)

    # Vẽ từng góc
    for pt, label, color in zip(sorted_pts.astype(int), labels, colors):
        cv2.circle(vis, tuple(pt), 12, color, -1)
        cv2.putText(vis, label, (pt[0] + 15, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Ghi phương pháp đã dùng
    method_names = {
        'contour': '1c. approxPolyDP',
        'hough': '1d. Hough Lines',
        'ml': '1e. YOLO Segmentation',
        'docaligner': '1e. DocAligner'
    }
    cv2.putText(vis, f"Method: {method_names.get(method, 'Unknown')}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return vis


def show_results(orig, result, use_hed=False):
    """Hiển thị kết quả Step 1 bằng matplotlib."""
    steps = [("Ảnh gốc", orig)]

    # Edges
    edge_title = "1b. HED Edges" if use_hed else "1b. Canny Edges"
    steps.append((edge_title, result['edged']))

    # ML mask (nếu có)
    if result['mask'] is not None:
        steps.append(("1e. ML Mask", result['mask']))

    # 4 góc
    if result['corners'] is not None:
        corners_vis = draw_corners(orig, result['corners'], result['method'])
        steps.append((f"Kết quả ({result['method']})", corners_vis))

    # Step 2: Unwarped
    if 'warped' in result and result['warped'] is not None:
        steps.append(("Step 2. Perspective Warp", result['warped']))

    # Step 3: Enhanced
    if 'enhanced' in result and result['enhanced'] is not None:
        steps.append(("Step 3. Enhanced output", result['enhanced']))

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

    plt.suptitle("Step 1 — Document Detection Pipeline", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def get_image_dir():
    """Trả về đường dẫn tới thư mục image/ của project."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image")


def list_categories():
    """Liệt kê các thư mục con trong image/."""
    image_dir = get_image_dir()
    categories = []
    for name in sorted(os.listdir(image_dir)):
        full = os.path.join(image_dir, name)
        if os.path.isdir(full) and not name.startswith('.'):
            count = len([f for f in os.listdir(full) if f.endswith(('.jpg', '.png', '.jpeg'))])
            if count > 0:
                categories.append((name, count))
    return categories


def get_images_in_category(category):
    """Lấy danh sách ảnh trong 1 thư mục con."""
    cat_dir = os.path.join(get_image_dir(), category)
    if not os.path.isdir(cat_dir):
        return []
    return sorted([
        os.path.join(cat_dir, f) for f in os.listdir(cat_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])


def main():
    import argparse
    """
    Entry point chạy thử toàn bộ Pipeline.
    
    Cách dùng:
        python main.py fold 4                 → Chạy ảnh fold góc 4
        python main.py list                   → Liệt kê tất cả thư mục ảnh
        python main.py --force-ml <...>       → Bỏ qua CV truyền thống, chạy thẳng ML
        python main.py --hed <...>            → Sử dụng mạng HED thay cho Canny
        python main.py --dewarp-ml <...>      → Kích hoạt ML Dewarping ở Step 2
    """

    # ── Parse flags ──
    parser = argparse.ArgumentParser(description="Document Detection Pipeline")
    parser.add_argument("input", nargs="?", help="Đường dẫn ảnh, tên thư mục, hoặc 'list'")
    parser.add_argument("index", nargs="?", type=int, default=0, help="Số thứ tự ảnh trong thư mục (mặc định 0)")
    parser.add_argument("--force-ml", action="store_true", help="Bỏ qua Canny/Hough, ép chạy thẳng Machine Learning (YOLOv8-Seg) ở Step 1")
    parser.add_argument("--docaligner", action="store_true", help="Sử dụng mô hình DocAligner (SoTA) chuyên dụng để tìm góc vuông văn bản ở Step 1 thay cho YOLO")
    parser.add_argument("--hed", action="store_true", help="Sử dụng mạng HED thay cho Canny Edge Detection")
    parser.add_argument("--dewarp-ml", action="store_true", help="Kích hoạt Document Dewarping bằng ML ở Step 2 (Là phẳng trang giấy cong vật lý)")
    args = parser.parse_args()

    # ── Xác định ảnh đầu vào ──
    if args.input:
        # Lệnh "list" → liệt kê thư mục
        if args.input == "list":
            print("📂 Các thư mục ảnh có sẵn:")
            for name, count in list_categories():
                print(f"   {name:15s}  ({count} ảnh)")
            print(f"\nCách dùng: python main.py <tên_thư_mục> [số_thứ_tự]")
            print(f"           python main.py --force-ml <tên_thư_mục>")
            print(f"           python main.py --hed <tên_thư_mục>")
            print(f"           python main.py --dewarp-ml <tên_thư_mục>")
            return

        # Nếu arg1 là đường dẫn file tồn tại → dùng luôn
        if os.path.isfile(args.input):
            image_path = args.input
        else:
            # Xem arg1 có phải tên thư mục con trong image/ không
            images = get_images_in_category(args.input)
            if images:
                idx = min(args.index, len(images) - 1)
                image_path = images[idx]
                print(f"📂 Thư mục: {args.input} ({len(images)} ảnh), chọn ảnh #{idx}")
            else:
                print(f"❌ Không tìm thấy ảnh hoặc thư mục: {args.input}")
                print(f"\nCác thư mục có sẵn:")
                for name, count in list_categories():
                    print(f"   {name:15s}  ({count} ảnh)")
                return
    else:
        # Mặc định: lấy ảnh đầu tiên của thư mục perspective
        default_category = "perspective"
        images = get_images_in_category(default_category)
        if images:
            image_path = images[0]
        else:
            print("❌ Không tìm thấy ảnh mẫu. Kiểm tra thư mục image/")
            return

    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        return

    # ── Đọc ảnh ──
    img = cv2.imread(image_path)
    basename = os.path.basename(image_path)
    category = os.path.basename(os.path.dirname(image_path))

    mode_label = []
    if args.docaligner:
        mode_label.append("🔴 DocAligner SOTA")
    elif args.force_ml:
        mode_label.append("🔴 YOLO ML ONLY")
    else:
        mode_label.append("🟢 CV + YOLO Fallback")
    if args.hed:
        mode_label.append("🔴 HED Edges")
    if args.dewarp_ml:
        mode_label.append("🔴 ML DEWARPING")
    
    print(f"═══════════════════════════════════════════")
    print(f"  STEP 1: Document Detection Pipeline")
    print(f"  Ảnh:      {category}/{basename}")
    print(f"  Kích thước: {img.shape[1]}×{img.shape[0]}")
    print(f"  Mode:     {', '.join(mode_label)}")
    print(f"═══════════════════════════════════════════\n")

    # ── Chạy pipeline (Hybrid) ──
    detector = DocumentDetector(enable_ml=True, use_hed=args.hed, use_docaligner=args.docaligner, use_ml_dewarp=args.dewarp_ml)

    # Nếu gọi theo Hybrid Force-ML (bỏ CV cơ bản)
    blurred, resized, ratio = detector.preprocessor.process(img)
    print(f"[1a] Tiền xử lý: {img.shape} → resize {resized.shape}  ✓")
    
    if args.docaligner:
        print(f"[1b] Model Segmentation (DocAligner - State of the Art)...")
        mask, corners = detector.docaligner_segmentor.segment(resized)
        if corners is None:
            # Fallback về YOLO
            print(f"[Thuộc lòng] DocAligner lỗi, Fallback về YOLOv8.")
            mask, corners = detector.ml_segmentor.segment(resized)
            method = 'ml' if corners is not None else None
        else:
            method = 'docaligner'
    else:
        print(f"[1b] ML Segmentation (Mặc định cho Hybrid Pipeline YOLOv8)...")
        mask, corners = detector.ml_segmentor.segment(resized)
        method = 'ml' if corners is not None else None

    result = {
        'corners': corners * ratio if corners is not None else None,
        'method': method,
        'edged': detector.hed_detector.detect(resized) if args.hed else detector.edge_detector.detect(blurred),
        'blurred': blurred,
        'resized': resized,
        'ratio': ratio,
        'mask': mask,
    }

    # ── Step 2: Perspective Transform / ML Dewarp ──
    print(f"\n═══════════════════════════════════════════")
    if result['corners'] is not None:
        print(f"\n[Step 2] Perspective Transform / Dewarping...")
        try:
            warped = detector.transformer.dewarp(img, result['corners']) if hasattr(detector.transformer, 'dewarp') else detector.transformer.transform(img, result['corners'])
            result['warped'] = warped
            print(f"  ✓ Đã căn chỉnh mặt phẳng: {warped.shape[1]}x{warped.shape[0]}")
            print(f"  ✅ Thành công! Phương pháp: {result['method']}")
            
            if len(result['corners']) == 4:
                sorted_corners = CornerSorter.sort(result['corners'])
                print(f"  TL: ({sorted_corners[0][0]:.0f}, {sorted_corners[0][1]:.0f})")
                print(f"  TR: ({sorted_corners[1][0]:.0f}, {sorted_corners[1][1]:.0f})")
                print(f"  BR: ({sorted_corners[2][0]:.0f}, {sorted_corners[2][1]:.0f})")
                print(f"  BL: ({sorted_corners[3][0]:.0f}, {sorted_corners[3][1]:.0f})")
            else:
                print(f"  ⚠️ Tọa độ trả về không đủ 4 điểm ({len(result['corners'])} điểm). Bỏ qua hiển thị góc.")
                
            # ── Step 3: Enhancement ──
            print(f"\n[Step 3] Enhancement (Shadow Removal & Binarization)...")
            enhanced = detector.enhancer.enhance(warped)
            result['enhanced'] = enhanced
            print(f"  ✓ Đã tăng cường chất lượng ảnh!")
            
        except Exception as e:
            print(f"  ❌ Lỗi transform hoặc enhancement: {e}")
        
    else:
        print(f"  ❌ Thất bại — Không tìm được 4 góc")
    print(f"═══════════════════════════════════════════")
    # ── Hiển thị ──
    if not args.force_ml:
        show_results(img, result, args.hed)


if __name__ == "__main__":
    main()
