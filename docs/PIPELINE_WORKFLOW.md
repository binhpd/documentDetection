# Document Scanner — Pipeline Workflow (Đầy đủ)

> 🟢 = Image Processing &nbsp;|&nbsp; 🔴 = Machine Learning

---

## STEP 1: PHÁT HIỆN VÙNG TÀI LIỆU (Document Detection)

### 1a. Resize ảnh 🟢 Image Processing
- **Làm gì:** Thu nhỏ ảnh gốc về chiều cao cố định (500px) để tăng tốc xử lý, lưu lại tỷ lệ `ratio` để quy đổi toạ độ về ảnh gốc sau này.
- **Input:** Ảnh màu BGR gốc từ camera (H × W × 3, uint8)
- **Output:** Ảnh đã resize (500 × W' × 3, uint8) + giá trị `ratio = H/500`

### 1b. Chuyển xám + Làm mịn 🟢 Image Processing
- **Làm gì:** Chuyển ảnh màu sang ảnh xám (1 kênh) rồi áp Gaussian Blur để loại bỏ nhiễu hạt, giúp bước phát hiện cạnh tiếp theo không bị nhầm nhiễu thành cạnh.
- **Input:** Ảnh màu đã resize (500 × W' × 3, uint8)
- **Output:** Ảnh xám đã làm mịn (500 × W' × 1, uint8)

### 1c. Phát hiện cạnh — Canny Edge Detection 🟢 Image Processing
- **Làm gì:** Tính gradient cường độ sáng tại mỗi pixel, giữ lại những pixel có gradient đột biến (cạnh). Sử dụng ngưỡng tự động dựa trên giá trị median để thay thế ngưỡng cứng (75, 200).
- **Input:** Ảnh xám đã blur (500 × W' × 1, uint8, giá trị 0-255)
- **Output:** Ảnh nhị phân cạnh (500 × W' × 1, uint8, giá trị chỉ 0 hoặc 255)

### 1d. Tìm 4 góc — Contour + approxPolyDP 🟢 Image Processing
- **Làm gì:** Tìm tất cả đường bao khép kín (contour) trên ảnh cạnh, sắp xếp theo diện tích giảm dần, lấy 5 contour lớn nhất. Với mỗi contour, xấp xỉ thành đa giác (Douglas-Peucker). Nếu đa giác có đúng 4 đỉnh → đó là viền tờ giấy.
- **Input:** Ảnh nhị phân cạnh (500 × W' × 1, 0/255)
- **Output:** Mảng 4 toạ độ góc `(4, 2) float32` — hoặc `None` nếu không tìm được

### 1e. Fallback — Hough Line Transform + Line Intersection 🟢 Image Processing
- **Khi nào chạy:** Chỉ khi bước 1d thất bại (trả về `None` — ví dụ góc bị tay che, viền đứt đoạn).
- **Làm gì:** Dùng biến đổi Hough tìm tất cả đoạn thẳng trên ảnh cạnh. Gom cụm các đoạn thẳng thành 4 nhóm (Trên/Dưới/Trái/Phải) dựa trên góc nghiêng và vị trí. Tính giao điểm toán học của 4 cặp đường để thu được 4 góc ảo (ngay cả khi góc vật lý bị che khuất).
- **Input:** Ảnh nhị phân cạnh (500 × W' × 1, 0/255)
- **Output:** Mảng 4 toạ độ giao điểm `(4, 2) float32` — hoặc `None` nếu không đủ thông tin

### 1f. Fallback cuối — Document Segmentation 🔴 Machine Learning
- **Khi nào chạy:** Chỉ khi cả bước 1d và 1e đều thất bại (nền quá phức tạp, giấy cùng tông màu với nền, hoặc giấy bị cong/nhăn không có cạnh thẳng nào).
- **Làm gì:** Dùng mạng nơ-ron (U-Net hoặc YOLOv8-Seg) đã được huấn luyện trên ảnh tài liệu để phân vùng ngữ nghĩa — mạng tô màu từng pixel: "là giấy" hoặc "không phải giấy". Sau đó dùng `minAreaRect` hoặc Hough Lines trên mask này để trích 4 góc.
- **Input:** Ảnh màu đã resize (500 × W' × 3, uint8)
- **Output:** Mask nhị phân vùng giấy (500 × W' × 1, 0/255) → trích ra 4 toạ độ góc `(4, 2) float32`

---

## STEP 2: BIẾN ĐỔI HÌNH HỌC (Geometric Transformation)

### 2a. Sắp xếp 4 góc (Corner Sorting) 🟢 Image Processing
- **Làm gì:** Sắp xếp 4 điểm lộn xộn thành thứ tự cố định [Top-Left, Top-Right, Bottom-Right, Bottom-Left] dựa trên tổng (x+y) và hiệu (y−x). Nhân toạ độ với `ratio` để quy về kích thước ảnh gốc.
- **Input:** 4 toạ độ góc chưa sắp xếp `(4, 2) float32`
- **Output:** 4 toạ độ đã sắp xếp `[TL, TR, BR, BL]` `(4, 2) float32` trên ảnh gốc

### 2b. Perspective Transform (Biến đổi phối cảnh) 🟢 Image Processing
- **Làm gì:** Tính ma trận biến đổi phối cảnh 3×3 ánh xạ 4 góc nghiêng → 4 góc chữ nhật. Áp dụng `warpPerspective` để "kéo phẳng" tờ giấy thành ảnh chữ nhật nhìn thẳng (bird's-eye view).
- **Input:** Ảnh gốc đầy đủ (H × W × 3, uint8) + 4 góc đã sắp xếp `(4, 2) float32`
- **Output:** Ảnh tài liệu đã duỗi phẳng, hình chữ nhật (maxH × maxW × 3, uint8)

### 2c. Document Dewarping (Tuỳ chọn) 🔴 Machine Learning
- **Khi nào chạy:** Khi phát hiện tài liệu bị cong/nhăn (trang sách mở, giấy bị gấp, giấy nhàu). Có thể phát hiện bằng cách kiểm tra độ cong của contour hoặc dùng heuristic.
- **Làm gì:** Dùng mạng nơ-ron (FDRNet / DewarpNet / DocUNet / GeoTr) dự đoán bản đồ pixel-wise remapping. Thay vì chỉ biến đổi tuyến tính 4 góc, mạng dự đoán hàng nghìn điểm điều khiển (control points) trên lưới dày đặc, sau đó dùng phép biến đổi Thin-Plate Spline (TPS) để duỗi phẳng từng vùng nhỏ một cách phi tuyến.
- **Input:** Ảnh tài liệu bị cong (maxH × maxW × 3, uint8)
- **Output:** Ảnh tài liệu đã duỗi phẳng hoàn toàn (H' × W' × 3, uint8)

---

## STEP 3: TĂNG CƯỜNG CHẤT LƯỢNG (Image Enhancement)

### 3a. Loại bỏ bóng đổ (Shadow Removal) 🟢 Image Processing
- **Làm gì:** Chuyển ảnh sang xám. Dùng Morphological Close với kernel lớn (21×21) để ước lượng "bản đồ ánh sáng nền" (mọi nét chữ bị xoá, chỉ còn gradient bóng). Sau đó chia ảnh gốc cho bản đồ nền → bóng đổ bị triệt tiêu.
- **Input:** Ảnh tài liệu đã warp (maxH × maxW × 3, uint8)
- **Output:** Ảnh xám không bóng, ánh sáng đồng đều (maxH × maxW × 1, uint8)

### 3b. Cân bằng tương phản — CLAHE 🟢 Image Processing
- **Làm gì:** Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization). Chia ảnh thành lưới 8×8 tile, mỗi tile cân bằng histogram riêng biệt để tăng tương phản cục bộ. Clip limit giới hạn mức khuếch đại để tránh quá sáng.
- **Input:** Ảnh xám đã loại bóng (maxH × maxW × 1, uint8)
- **Output:** Ảnh xám tương phản cao, chữ đen rõ trên nền trắng (maxH × maxW × 1, uint8)

### 3c. Nhị phân hoá — Adaptive Thresholding 🟢 Image Processing
- **Làm gì:** Chuyển ảnh xám thành ảnh 2 màu (đen/trắng). Mỗi vùng nhỏ (blockSize × blockSize pixel) tự tính ngưỡng riêng dựa trên trung bình Gaussian có trọng số, trừ đi hằng số C. Pixel sáng hơn ngưỡng → trắng (nền), tối hơn → đen (chữ).
- **Input:** Ảnh xám tương phản cao (maxH × maxW × 1, uint8, giá trị 0-255)
- **Output:** Ảnh nhị phân (maxH × maxW × 1, uint8, giá trị chỉ 0 hoặc 255)

### 3d. Làm sạch hình thái học — Morphological Cleaning 🟢 Image Processing
- **Làm gì:** Opening (erode → dilate): Xoá các chấm nhiễu nhỏ hơn kernel. Closing (dilate → erode): Lấp các lỗ hổng nhỏ trong nét chữ. Kết quả là ảnh scan sạch, nét chữ liền mạch.
- **Input:** Ảnh nhị phân có nhiễu (maxH × maxW × 1, 0/255)
- **Output:** ✅ **Ảnh tài liệu sạch cuối cùng** (maxH × maxW × 1, 0/255)

### 3e. Deep Binarization (Tuỳ chọn, thay thế 3a-3d) 🔴 Machine Learning
- **Khi nào chạy:** Khi tài liệu bị suy thoái nặng (ố vàng, nhàu nát, chữ viết tay mờ, watermark, stain). Adaptive Threshold không phân biệt được "vết bẩn" với "chữ".
- **Làm gì:** Dùng mạng nơ-ron (DocEnTr — Vision Transformer, hoặc DE-GAN — Generative Adversarial Network) đã được huấn luyện trên dataset DIBCO. Mạng học cách nhận biết "đâu là chữ" ở mức ngữ nghĩa, bỏ qua vết bẩn, bóng, watermark. Thay thế toàn bộ 3a→3d bằng 1 bước duy nhất.
- **Input:** Ảnh xám tài liệu (maxH × maxW × 1, uint8)
- **Output:** Ảnh nhị phân sạch (maxH × maxW × 1, 0/255)

---

## Tổng kết

| Phân loại | Số bước | Các bước | Vai trò |
|---|---|---|---|
| 🟢 **Image Processing** | **10 bước** | 1a, 1b, 1c, 1d, 1e, 2a, 2b, 3a, 3b, 3c, 3d | **Pipeline chính** — hoạt động độc lập, không cần ML |
| 🔴 **Machine Learning** | **3 bước** | 1f, 2c, 3e | **Fallback / Tuỳ chọn** — chỉ bật khi Image Processing thất bại |

> Pipeline **mặc định chạy hoàn toàn bằng Image Processing**. Các bước Machine Learning là **tuỳ chọn nâng cao**, chỉ kích hoạt khi gặp điều kiện ảnh cực đoan mà thuật toán truyền thống không xử lý được.
