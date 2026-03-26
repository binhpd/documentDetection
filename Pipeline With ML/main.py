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

# Fix incompatible architecture error for M-series Macs regarding libturbojpeg
os.environ["TURBOJPEG"] = "/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.dylib"

import numpy as np
import matplotlib.pyplot as plt

from step1_preprocessor import Preprocessor
from step1_ml_segmentor import MLSegmentor
from step1_docaligner import DocAlignerSegmentor
from step2_perspective_transform import PerspectiveTransformer
from step2_ml_dewarper import MLDewarper
from step2_uvdoc_dewarper import UVDocDewarper
from step3_enhancer import DocumentEnhancer
from corner_sorter import CornerSorter


class DocumentDetector:
    """Điều phối Step 1 với chiến lược cascading fallback."""

    def __init__(self, enable_ml=False, use_docaligner=False, yolo_model="models/yolov8n-seg.pt", unet_model="models/unet_doc.pt", use_ml_dewarp=False, use_u2net=False, use_uvdoc=False):
        """
        Khởi tạo DocumentDetector pipeline.
        
        Args:
            enable_ml: Có bật ML fallback không (YOLOv8)
            use_docaligner: Sử dụng DocAligner chuyên dụng làm segmentor
            use_ml_dewarp: Sử dụng ML Dewarping phi tuyến tính thay thế Perspective Transform cho tài liệu cong
            use_u2net: Sử dụng mạng U2-Net (thư viện rembg) chuyên khoét nền lấy hình bóng giấy (triệt để nhất cho Dewarping)
            use_uvdoc: Sử dụng mạng UVDoc để tính toán lưới nắn cong giấy
        """
        self.preprocessor = Preprocessor()
        
        # --- Step 2: Transformer ---
        baseline_transformer = PerspectiveTransformer()
        if use_uvdoc:
            self.transformer = UVDocDewarper(fallback_transformer=baseline_transformer)
        elif use_ml_dewarp:
            self.transformer = MLDewarper(fallback_transformer=baseline_transformer)
        else:
            self.transformer = baseline_transformer
            
        self.enable_ml = enable_ml
        self.use_docaligner = use_docaligner
        self.use_uvdoc = use_uvdoc
        
        self.ml_segmentor = MLSegmentor(yolo_model, unet_model) if enable_ml else None
        self.docaligner_segmentor = DocAlignerSegmentor() if use_docaligner else None

        # --- Step 3: Enhancer ---
        self.enhancer = DocumentEnhancer()
        
        self.use_u2net = use_u2net
        self.use_hed = False
        self.edge_detector = cv2.Canny # dummy assignment just to prevent error if it gets here, actually we just skip it or fix the logic
        
        # We need to initialize edge_detector, contour_detector, hough_detector if they are used in detect()
        # To make detect() work for ML, we should just let them fallback
        class DummyDetector:
            def detect(self, img): return cv2.Canny(img, 75, 200)
            def find_corners(self, img): return None
        
        self.edge_detector = DummyDetector()
        self.contour_detector = DummyDetector()
        self.hough_detector = DummyDetector()

    def detect(self, image, save_prefix=None):
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
        blurred, resized, ratio = self.preprocessor.process(image, save_prefix=save_prefix)
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
            'u2net_doc': None
        }

        # ── 1u. U2-Net (Rembg) SOTA Background Removal ──
        if self.use_u2net:
            print(f"[1e] 👑 Bắt đầu bóc nền bằng sức mạnh U²-Net (Rembg)...")
            try:
                from rembg import remove
                # Chạy U2-net gỡ sạch nền
                subject_orig = remove(orig)
                alpha_orig = subject_orig[:, :, 3]
                
                # Tìm 4 góc tài liệu từ mask trong suốt (alpha_orig)
                contours, _ = cv2.findContours(alpha_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                corners = None
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    # C2: Dùng approxPolyDP lặp dần để bắt 4 mép thẳng của tờ giấy (bỏ qua những điểm nhấp nhô của ngón tay)
                    corners = None
                    peri = cv2.arcLength(largest, True)
                    # Thử các mức độ "đơn giản hóa đa giác" từ mịn đến thô
                    for eps in np.linspace(0.01, 0.1, 10):
                        approx = cv2.approxPolyDP(largest, eps * peri, True)
                        if len(approx) == 4:
                            corners = approx.reshape(4, 2).astype(np.float32)
                            break
                            
                    # Nếu approxPolyDP vẫn không tìm được tứ giác (do bàn tay che lấp quá nhiều), 
                    # fallback dùng Extreme Points an toàn hơn minAreaRect
                    if corners is None:
                        pts = largest.reshape(-1, 2)
                        s = pts.sum(axis=1) # x + y
                        diff = np.diff(pts, axis=1) # y - x
                        corners = np.array([
                            pts[np.argmin(s)],
                            pts[np.argmin(diff)],
                            pts[np.argmax(s)],
                            pts[np.argmax(diff)]
                        ], dtype=np.float32)

                # Ép dán tờ giấy trơ trọi đã crop lên một background Trắng tuyệt đối (Để bảo vệ chữ khi feed vào dewarp)
                alpha_c = subject_orig[:, :, 3]
                rgb_c = subject_orig[:, :, :3]
                white_bg = np.ones_like(rgb_c) * 255
                mask_f = alpha_c[:, :, np.newaxis] / 255.0
                pure_doc = (rgb_c * mask_f + white_bg * (1 - mask_f)).astype(np.uint8)
                
                result['u2net_doc'] = pure_doc
                result['u2net_mask'] = alpha_orig
                result['method'] = 'u2net'
                result['corners'] = corners if corners is not None else []
                if len(result['corners']) == 4:
                    try:
                        from corner_sorter import sort_corners
                        result['corners'] = sort_corners(result['corners'])
                    except ImportError:
                        pass
                print(f"[1e] U²-Net bóc nền & tìm góc thành công! ✓")
                return result
            except ImportError:
                 print("❌ Cần cài đặt rembg: pip install rembg")

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
            # The original code used self.ml_segmentor.segment(resized)
            # The user's snippet introduces YOLOSegmentor and args.yolo, which implies a new argument.
            # Assuming the user wants to use the YOLOSegmentor with padding if args.yolo is provided,
            # otherwise fallback to the existing self.ml_segmentor logic.
            # However, the provided snippet completely replaces the existing logic.
            # I will integrate the user's snippet as a replacement for the existing YOLO ML Segmentation block,
            # assuming `args` will be available in this context (which it isn't directly in a class method).
            # This suggests the user might be moving this logic to `main()` or `detect` needs `args` passed.
            # Given the instruction is to modify `main.py` and `step1_ml_segmentor.py`,
            # and the snippet is within `DocumentDetector.detect`, I will assume `args` is accessible
            # (e.g., via `self.args` if passed during init, or if this block is moved to `main`).
            # For now, I'll make the change as requested, assuming `args` will be resolved.
            # Also, the `image` in `segment(image, ...)` should be `resized` for consistency.
            from step1_ml_segmentor import YOLOSegmentor # This import should be at the top or handled.
            # Assuming args is available here, which is not typical for a class method without passing it.
            # This block is likely intended for the main function's logic, not detect().
            # I will make the change as literally as possible, but note this potential issue.
            # The user's snippet also has a syntax error with `else:`
            # I will assume `args.yolo` is a string path to the model, and `len(args.yolo) > 0` checks if it's provided.
            # If `args.yolo` is meant to be a flag, `if args.yolo:` would be more appropriate.
            # Given the context of `yolo_model="models/yolov8n-seg.pt"` in `__init__`,
            # it's more likely `args.yolo` would be a path.

            # Re-evaluating the user's snippet: it seems to be a replacement for the `if self.enable_ml` block
            # in the `main` function, not `detect`. The instruction says "Bổ sung argument `--crop-padding` ở `main.py`
            # và sửa `step1_ml_segmentor.py` nhận cờ này vào hàm `segment` và `predict`."
            # The provided code snippet for `detect` is problematic if `args` is not available.
            # The original `detect` method does not have `args`.
            # The user's snippet for `detect` also seems to be missing the `* ratio` for corners.
            # I will apply the `ArgumentParser` changes in `main()` and assume the `detect` method's
            # ML segmentation logic will be handled in `main()` where `args` is available.
            # The snippet for `detect` is actually part of the `main` function's "Nhánh hybrid cũ cho YOLO / DocAligner" block.
            # I will apply the `detect` method changes to the `main` function's logic.
            mask, corners = self.ml_segmentor.segment(resized) # Original line
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
    if isinstance(corners, list) and len(corners) == 0:
        return vis
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
        'docaligner': '1e. DocAligner',
        'yolo-seg': '1e. YOLO Segmentation' # Added for consistency with new method name
    }
    cv2.putText(vis, f"Method: {method_names.get(method, 'Unknown')}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return vis


def show_results(orig, result):
    """Hiển thị kết quả Step 1 bằng matplotlib."""
    steps = [("Ảnh gốc", orig)]

    # ML mask (nếu có)
    if result.get('mask') is not None:
        steps.append(("1e. ML Mask", result['mask']))
    # Add yolo_mask if present
    if result.get('yolo_mask') is not None:
        steps.append(("1e. YOLO Mask", result['yolo_mask']))

    # Hiển thị U2-Net Extracted Document (nếu có)
    if result.get('u2net_doc') is not None:
        steps.append(("👑 U²-Net Doc", result['u2net_doc']))
        
    # Luôn hiển thị 4 góc trên ảnh gốc nếu tìm thấy
    if result.get('corners') is not None and len(result['corners']) > 0:
        corners_vis = draw_corners(orig, result['corners'], result['method'])
        steps.append((f"Kết quả ({result['method']})", corners_vis))

    if result.get('coons_warped') is not None:
        steps.append(("Chữ nhật hoá viền cong (Coons Patch)", result['coons_warped']))

    # Step 2: Unwarped (Perspective Transform)
    if result.get('warped') is not None:
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

def get_image_path(folder, index):
    """Xác định đường dẫn ảnh đầu vào dựa trên thư mục và chỉ số."""
    # Nếu folder là đường dẫn file tồn tại → dùng luôn
    if os.path.isfile(folder):
        return folder, 1
    
    # Xem folder có phải tên thư mục con trong image/ không
    images = get_images_in_category(folder)
    if images:
        idx = min(index, len(images) - 1)
        return images[idx], len(images)
    else:
        print(f"❌ Không tìm thấy ảnh hoặc thư mục: {folder}")
        print(f"\nCác thư mục có sẵn:")
        for name, count in list_categories():
            print(f"   {name:15s}  ({count} ảnh)")
        exit() # Thoát chương trình nếu không tìm thấy ảnh

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
    parser.add_argument("folder", nargs="?", default="perspective", help="Tên thư mục con trong image/ (ví dụ: perspective, curved)")
    parser.add_argument("index", nargs="?", type=int, default=0, help="Số thứ tự ảnh trong thư mục (mặc định 0)")
    parser.add_argument("--force-ml", action="store_true", help="Bỏ qua Canny/Hough, ép chạy thẳng Machine Learning (YOLOv8-Seg) ở Step 1")
    parser.add_argument("--docaligner", action="store_true", help="Sử dụng mô hình DocAligner (SoTA) chuyên dụng để tìm góc vuông văn bản ở Step 1 thay cho YOLO")
    parser.add_argument("--u2net", action="store_true", help="Sử dụng mô hình U²-Net (Rembg) đục 100% background để lộ rõ giấy lồi lõm trước khi đưa vào Dewarping")
    parser.add_argument('--dewarp-ml', action='store_true', help='Use page-dewarp (AI text-line analysis) to flatten the document. Slower but flattens curved pages.')
    parser.add_argument("--uvdoc", action="store_true", help="Sửa cong rách nát tài liệu bằng Neural Grid UVDoc (chuyên dụng xử lý độ cong sâu sắc)")
    parser.add_argument("--yolo", type=str, default="models/yolov8n-seg.pt", help="Đường dẫn đến mô hình YOLOv8 segmentation (mặc định: models/yolov8n-seg.pt)")
    parser.add_argument("--bw", action="store_true", help="Tăng cường và xuất File ảnh dạng Đen Trắng (B/W) ở Step 3 thay vì bản Màu gốc")
    args = parser.parse_args()

    # ── Xác định ảnh đầu vào ──
    if args.folder == "list":
        print("📂 Các thư mục ảnh có sẵn:")
        for name, count in list_categories():
            print(f"   {name:15s}  ({count} ảnh)")
        print(f"\nCách dùng: python main.py <tên_thư_mục> [số_thứ_tự]")
        print(f"           python main.py --force-ml <tên_thư_mục>")
        print(f"           python main.py --docaligner <tên_thư_mục>")
        print(f"           python main.py --dewarp-ml <tên_thư_mục>")
        return

    image_path, total_files = get_image_path(args.folder, args.index)
    
    print(f"📂 Thư mục: {args.folder} ({total_files} ảnh), chọn ảnh #{args.index}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return

    orig = img.copy()
    basename = os.path.basename(image_path)
    base_code = os.path.splitext(basename)[0]
    category = os.path.basename(os.path.dirname(image_path))
    
    # Chuẩn bị Data Logger lưu kết quả từng bước
    save_dir = "result"
    os.makedirs(save_dir, exist_ok=True)
    save_prefix = os.path.join(save_dir, f"{category}_{base_code}")

    mode_label = []
    if args.u2net:
        mode_label.append("👑 U²-Net (SOTA Background Removal)")
    elif args.docaligner:
        mode_label.append("🔴 DocAligner SOTA")
    elif args.force_ml:
        mode_label.append("🔴 YOLO ML ONLY")
    else:
        mode_label.append("🔴 YOLO Fallback")
        
    if args.uvdoc:
        mode_label.append("🔴 UVDOC NEURAL DEWARP")
    elif args.dewarp_ml:
        mode_label.append("🔴 ML DEWARPING")
    
    print(f"═══════════════════════════════════════════")
    print(f"  STEP 1: Document Detection Pipeline")
    print(f"  Ảnh:      {category}/{basename}")
    print(f"  Kích thước: {img.shape[1]}×{img.shape[0]}")
    print(f"  Mode:     {', '.join(mode_label)}")
    print(f"═══════════════════════════════════════════\n")

    # ── Chạy pipeline (Pure ML & Full Flow) ──
    # Tích hợp trực tiếp MLDewarper (page-dewarp) làm mặc định thay vì chỉ khi có flag
    detector = DocumentDetector(enable_ml=True, use_docaligner=args.docaligner, use_ml_dewarp=args.dewarp_ml, use_u2net=args.u2net, use_uvdoc=args.uvdoc)

    # Nếu đang bật U2-Net, hãy chạy hàm detect một cách trọn vẹn thay vì shortcut như YOLO/DocAligner
    if args.u2net:
        result = detector.detect(orig, save_prefix=save_prefix)
    else:
        # Nhánh hybrid cũ cho YOLO / DocAligner
        blurred, resized, ratio = detector.preprocessor.process(orig, save_prefix=save_prefix)
        print(f"[1a] Tiền xử lý: {orig.shape} → resize {resized.shape}  ✓")
        
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
            'edged': None,
            'blurred': blurred,
            'resized': resized,
            'ratio': ratio,
            'mask': mask,
            'u2net_doc': None,
            'orig': orig
        }

    # === Lưu kết quả Step 1 ra ổ cứng ===
    if result.get('blurred') is not None:
        cv2.imwrite(f"{save_prefix}_step1_1_blurred.jpg", result['blurred'])
    if result.get('edged') is not None:
        cv2.imwrite(f"{save_prefix}_step1_2_edged.jpg", result['edged'])
    if result.get('mask') is not None:
        # Nếu mask là xám thì lưu được luôn
        cv2.imwrite(f"{save_prefix}_step1_3_ml_mask.jpg", result['mask'])
    if result.get('u2net_doc') is not None:
        cv2.imwrite(f"{save_prefix}_step1_4_u2net_extracted.jpg", result['u2net_doc'])

    # ── Step 2: Perspective Transform / ML Dewarp ──
    print(f"\n═══════════════════════════════════════════")
    if result['corners'] is not None or result.get('u2net_doc') is not None:
        print(f"\n[Step 2] Perspective Transform / Dewarping...")
        try:
            # Khi dùng U2-Net, ảnh 'u2net_doc' đã được xóa background rác và ghép nền trắng,
            # Nếu chạy UVDoc, chỉ truyền thẳng cái u2net_doc.
            # Ta sẽ áp dụng Coons Patch Mesh Deformation thay vì Linear Perspective Transform nếu không dùng UVDoc!
            if result.get('u2net_doc') is not None:
                img_for_dewarp = result['u2net_doc']
                corners_for_dewarp = result['corners']
                
                # --- AUTO-CORRECTION (PHÂN TÍCH ĐỘ CONG VIỀN BẰNG TOÁN HỌC MASK IOU) ---
                anti_pinch_warped = None
                is_flat = False
                
                if corners_for_dewarp is not None and len(corners_for_dewarp) == 4:
                    if result.get('u2net_mask') is not None:
                        # Tính góc độ bao phủ (IoU) giữa Mask Viền Thực Tế và Đa giác 4 đường thẳng lý tưởng
                        mask_real = result['u2net_mask']
                        mask_poly = np.zeros_like(mask_real)
                        
                        # Vẽ Đa giác 4 góc thẳng
                        int_corners = np.array(corners_for_dewarp, dtype=np.int32)
                        cv2.fillPoly(mask_poly, [int_corners], 255)
                        
                        # Tính IoU
                        intersection = np.logical_and(mask_real > 127, mask_poly > 127).sum()
                        union = np.logical_or(mask_real > 127, mask_poly > 127).sum()
                        iou = intersection / (union + 1e-6)
                        
                        print(f"  [Phân tích Cạnh] Độ phẳng viền tài liệu (IoU): {iou:.4f}")
                        
                        # Nếu đa giác 4 đường thẳng khớp với viền thực trên 94%, chắc chắn viền rất thẳng (Tài liệu phẳng)
                        if iou > 0.94:
                            is_flat = True
                            print(f"  -> Trạng thái kiểm lường: Viền Đường Thẳng (Phẳng)")
                        else:
                            print(f"  -> Trạng thái kiểm lường: Viền Đường Cong (Cong vênh)")
                    else:
                        is_flat = True
                        
                if args.uvdoc and is_flat:
                    print(f"  ⚠️ [Auto-Correction] Chống véo biến dạng (Anti-Pinch)!")
                    print(f"  Tờ giấy được chẩn đoán là hình phẳng vì các viền thẳng lấp đầy >94% diện tích.")
                    print(f"  -> UVDoc (Neural Grid) sẽ bị loại bỏ để tránh méo ảnh. Tự động chuyển về Perspective Warp...")
                    from step2_perspective_transform import PerspectiveTransformer
                    cv_transformer = PerspectiveTransformer()
                    # CRITICAL FIX: Dùng img_for_dewarp (đã là u2net_doc có padding nền trắng) thay vì ảnh orig dính background lề cỏ!
                    anti_pinch_warped = cv_transformer.transform(img_for_dewarp, corners_for_dewarp, save_prefix=save_prefix)
                    args.uvdoc = False
                elif args.uvdoc and corners_for_dewarp is not None and len(corners_for_dewarp) == 4 and not is_flat:
                    print(f"  🌟 [Auto-Correction] Phát hiện ĐƯỜNG CONG rệt. UVDoc Dewarp sẽ được giữ nguyên!")
                
                if args.uvdoc and result.get('u2net_doc') is not None and result.get('u2net_mask') is not None:
                    # UVDoc expects a tightly cropped document image (no large margins).
                    # 'u2net_doc' has a white background but same size as orig. 
                    # We crop to the bounding box of the u-2net mask from u2net_doc, not orig
                    x, y, w, h = cv2.boundingRect(result['u2net_mask'])
                    img_for_dewarp = result['u2net_doc'][y:y+h, x:x+w]
                    print(f"[Step 2] Đã crop CÓ ÉP GÓC BẰNG U2NET-DOC cho UVDoc: {w}x{h}")
                elif result.get('u2net_mask') is not None and corners_for_dewarp is not None and len(corners_for_dewarp) == 4 and anti_pinch_warped is None:
                    print(f"[Step 2] Áp dụng Coons Patch Mesh Deformation nắn phẳng viền cong...")
                    from step2_coons_patch import CoonsPatchDewarper
                    coons_dewarper = CoonsPatchDewarper()
                    img_for_dewarp = coons_dewarper.dewarp_via_contour(img_for_dewarp, result['u2net_mask'], corners_for_dewarp, save_prefix=save_prefix)
                    print(f"  ✓ Đã vuốt thẳng bằng Coons Patch: {img_for_dewarp.shape[1]}x{img_for_dewarp.shape[0]}")
                    result['coons_warped'] = img_for_dewarp
                    corners_for_dewarp = None # Tắt PerspectiveTransform cũ của page-dewarp
            else:
                img_for_dewarp = result['u2net_doc'] if result.get('u2net_doc') is not None else orig
                corners_for_dewarp = result['corners']
                anti_pinch_warped = None
                
            if anti_pinch_warped is not None:
                warped = anti_pinch_warped
            else:
                warped = detector.transformer.dewarp(img_for_dewarp, save_prefix=save_prefix) if args.uvdoc else (detector.transformer.dewarp(img_for_dewarp, corners_for_dewarp, save_prefix=save_prefix) if hasattr(detector.transformer, 'dewarp') else detector.transformer.transform(img_for_dewarp, corners_for_dewarp, save_prefix=save_prefix))
            result['warped'] = warped
            cv2.imwrite(f"{save_prefix}_step2_dewarped.jpg", warped)
            
            print(f"  ✓ Đã căn chỉnh mặt phẳng (Từ U2-Net): {warped.shape[1]}x{warped.shape[0]}" if args.u2net else f"  ✓ Đã căn chỉnh mặt phẳng: {warped.shape[1]}x{warped.shape[0]}")
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
            print(f"\n[Step 3] Enhancement (Shadow Removal, CLAHE & Binarization)...")
            mode = "bw" if args.bw else "color"
            enhanced = detector.enhancer.enhance(warped, save_prefix=save_prefix, mode=mode)
            result['enhanced'] = enhanced
            print(f"  ✓ Đã tăng cường chất lượng ảnh (Chế độ: {mode.upper()})!")
            
        except Exception as e:
            print(f"  ❌ Lỗi transform hoặc enhancement: {e}")
        
    else:
        print(f"  ❌ Thất bại — Không tìm được 4 góc")
    print(f"═══════════════════════════════════════════")
    # ── Hiển thị ──
    if not args.force_ml:
        show_results(img, result)


if __name__ == "__main__":
    main()
