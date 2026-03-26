# Phân tích các Thách thức Vật lý khi Chụp ảnh Tài liệu & Vấn đề Tài liệu có Hình ảnh (Màu)

Tài liệu này tổng hợp 20 thách thức vật lý cốt lõi khi dùng Smartphone để quét (scan) tài liệu, đồng thời bổ sung phân tích chuyên sâu về trường hợp tài liệu chứa hình ảnh/đồ họa màu.

---

## PHẦN 1: BẢNG CHECKLIST 20 ISSUE KHI QUÉT TÀI LIỆU (ĐỐI CHIẾU PIPELINE HIỆN TẠI)

Dưới đây là 20 vấn đề được chia thành 4 nhóm chính, cùng với thực trạng xử lý của hệ thống `Pipeline With ML` hiện tại:

| STT | Vấn đề / Thách thức vật lý | Tình trạng | Giải pháp / Thuật toán đã áp dụng |
| :--- | :--- | :---: | :--- |
| **Nhóm 1: Giao thoa Ánh sáng rắc rối (Illumination & Lighting Issues)** |
| 1 | Đổ bóng bàn tay/ngón tay lên văn bản | ✅ Đã xử lý | **Division-based Shadow Normalization** (Step 3c) |
| 2 | Bóng đen nguyên khối của điện thoại in giữa giấy | ✅ Đã xử lý | **Division-based Shadow Normalization** (Step 3c) |
| 3 | Độ rọi tản sáng (Gradient) không đều / nửa lóa nửa râm | ✅ Đã xử lý | **Division-based Shadow Normalization** (Step 3c) |
| 4 | Lóa sáng dải dài dọc biên mép do đèn huỳnh quang | ✅ Đã xử lý | **Shadow Normalization** + **Adaptive Binarization** (Step 3c, 3d) |
| 5 | Đốm lóa trắng xóa (Glare) che khuất chữ do bật Flash | ✅ Đã xử lý | **In-painting** (Xóa đốm và nội suy vùng ảnh lân cận) (Step 3a) |
| 6 | Nhiễu hạt (ISO Noise / Salt-Pepper) do chụp thiếu sáng | ✅ Đã xử lý | **Adaptive Binarization** (Kích trần khoảng trắng che lấp nhiễu râm) (Step 3d) |
| 7 | Sai lệch phổ màu trắng do môi trường ám ánh đèn vàng/đỏ | ✅ Đã xử lý | **Grayscale / LAB** kết hợp **Adaptive Histogram Percentile** (Step 3d) |
| **Nhóm 2: Biến dạng Hình học & Không gian (Geometric & Spatial Distortions)** |
| 8 | Lệch góc phối cảnh (Perspective Distortion) do tư thế cầm | ✅ Đã xử lý | **ML Segmentor (DocAligner/U²-Net)** + **Perspective Transform** (Step 1, 2a) |
| 9 | Rìa góc giấy gập gãy, quăn mép, gió thổi bay | ✅ Đã xử lý | **U²-Net (Rembg)** bóc tách nền linh động bảo toàn nếp lượn sóng (Luồng A) |
| 10 | Biến dạng lượn cong phi tuyến tính do vùng gáy sách dày | ✅ Đã xử lý | **Text-line Dewarping** / **UVDoc Neural Grid** (Step 2b, 2c) |
| 11 | Bề mặt nhấp nhô lồi lõm do vò nát nhầu nhĩ | ✅ Đã xử lý | **U²-Net** bóc nền + **UVDoc** nắn phẳng 3D / Spline |
| 12 | Quang sai vật lý do thấu kính cụm cam góc rộng (Móp rìa) | ❌ Chưa xử lý | Chưa có **Camera Calibration** |
| **Nhóm 3: Sai hỏng Tiêu cự & Rung máy (Camera Hardware Constraints)** |
| 13 | Rung mờ chuyển động (Motion Blur) do thao tác tay bấm | ✅ Đã xử lý | **Unsharp Masking / Deblurring** (Step 3b) |
| 14 | Nhòe do mất tâm lấy nét tự động (Out-of-focus Blur) | ✅ Đã xử lý | **Unsharp Masking / Deblurring** (Step 3b) |
| 15 | Chụp xa hụt Crop dẫn tới phân giải hạt ảnh cực thấp | ❌ Chưa xử lý | Chưa có kiến trúc **Super Resolution / AI Upscaling** |
| 16 | Các dòng kẻ lưới nền bị dính nhiễu răng cưa (Moire pattern) | ❌ Chưa xử lý | Chưa có **FFT (Fast Fourier Transform)** để lọc dải tần số lặp |
| **Nhóm 4: Sự xuống cấp nội tạng Tài liệu (Document Surface Degradations)** |
| 17 | Nền giấy cũ mốc ngả vàng, ố bẩn lốm đốm / cặn bề mặt | ✅ Đã xử lý | **Adaptive Binarization** (Xén ngưỡng White Point tự động tẩy nền) (Step 3d) |
| 18 | Thấm, hằn rãnh mực mặt sau lên giấy mỏng (Bleed-through)| ⚠️ Một phần | Xén ngưỡng **White Point** tấy xóa gợn nhạt mặt sau (Step 3d) |
| 19 | Mực phai trôi, nhạt nhòa, vạch đứt đoạn không liền mạch | ✅ Đã xử lý | **Adaptive CLAHE** (Khóa chặt lõi mực phai thành đen sậm) (Step 3d) |
| 20 | Viết tay ngoằn ngoèo, dấu mộc đỏ in đè lấp văn bản đen | ❌ Chưa giải quyết| Con dấu đỏ bị hóa thành bóng xám lem nhem do thuật toán Grayscale. Chưa có **Color Separation** |

---

## PHẦN 2: PHÂN TÍCH MỞ RỘNG - KHI TÀI LIỆU CÓ CHỨA HÌNH ẢNH MÀU

Khi bức ảnh chụp không chỉ toàn văn bản (như hợp đồng, hóa đơn) mà là một trang tạp chí, bìa sách, sách giáo khoa hay báo chí **có chứa hình ảnh minh họa (màu)**, bài toán Scanner lúc này sẽ đối diện với **4 vấn đề lớn mới** cực kỳ nghiêm trọng:

### 1. Binarization "Phá hủy" Hình Ảnh (The Image Destruction Problem)
- **Vấn đề:** Các thuật toán kéo tương phản (như Soft Binarization) hay nhị phân hóa (Adaptive Thresholding) sinh ra để biến nền thành trắng tinh và chữ thành đen đậm. Tuy nhiên, nếu áp dụng lên một bức hình phong cảnh hoặc khuôn mặt người trong trang giấy, nó sẽ bóp nghẹt toàn bộ dải màu (Color depth) và Gradient ánh sáng của ảnh.
- **Hậu quả:** Bức ảnh màu minh họa sẽ bị biến thành những mảng đen trắng rỗ, chói lóa, mất đi định dạng hoàn toàn (như ảnh in lụa rẻ tiền).

### 2. Sự thay đổi Màu Thuần của Hình Ảnh do khử bóng (Color Shift via Shadow Removal)
- **Vấn đề:** Hàm Division-based Shadow Removal (khử bóng) hoạt động bằng cách mài mòn chi tiết để thu được một "Bề mặt ánh sáng nền". Khi áp vào một bức ảnh nhiều màu, thuật toán có thể hiểu nhầm những mảng màu tối (như bầu trời đêm) của hình minh họa là "Bóng đổ gắt" và tự động làm sáng rực nó lên một cách giả tạo.
- **Hậu quả:** Bức ảnh bị bạc màu, sai lệch mức phơi sáng, và các mảng khối trong hình minh họa bị đẩy sáng sai lệch so với bản gốc.

### 3. Nhiễu Moire Của Ảnh In Nhãn (Halftone Moire effect)
- **Vấn đề:** Hình ảnh in trên sách báo thường dùng mật độ các chấm mực siêu nhỏ (Halftone dots). Khi mắt kính điện thoại chụp lại, các lưới chấm này giao thoa với lưới phân giải của Digital Sensor, tạo ra các vạch sóng dập dềnh nhiều màu (như cầu vồng nhiễu), được gọi là hiện tượng Moire.
- **Hậu quả:** Hình minh họa quét được sẽ bị hằn lưới vân sóng, làm giảm mạnh thẩm mỹ của bản sách điện tử.

### 4. Vấn Đề Về Kích Thước Lưu Trữ PDF (File Size Bloat)
- **Vấn đề:** Mọi file scan đen/trắng truyền thống thường nén JBIG2 cực kỳ nhẹ (chỉ khoảng 50KB/trang). Nhưng nếu bảo toàn màu của bức ảnh, trang tài liệu không thể nén kiểu Binarize nữa mà phải nén full-color (JPEG). 
- **Hậu quả:** Nếu giữ toàn bộ trang giấy dưới dạng quét màu, dung lượng file PDF có thể phình to lên 2-5MB mỗi trang.

---

### 🔥 ĐỀ XUẤT GIẢI PHÁP CHUYÊN SÂU: KỸ THUẬT PHÂN TÁCH LỚP (Layout Analysis & Segmentation)

Để giải quyết bài toán tài liệu hỗn hợp (Văn bản + Ảnh), Hệ thống Scanner cao cấp không thể áp dụng chung 1 bộ lọc cho toàn bộ màn hình. Bắt buộc phải triển khai thuật toán **Document Layout Analysis (DLA)** theo các bước:

1. **Phân vùng (Segmentation):**
   - Sử dụng Mô hình trí tuệ nhân tạo (VD: `LayoutLM`, `U-Net Document Layout`) để vẽ bounding-box khoanh vùng riêng biệt: Đâu là **Text (Chữ)**, đâu là **Figure/Picture (Hình ảnh màu)**, đâu là **Background (Nền giấy trắng)**.

2. **Xử lý Chọn lọc (Selective Processing):**
   - **Với vùng TEXT (Chữ):** Áp dụng toàn bộ Bước 3 (Khử bóng, Phơi Sáng Tự Thích Ứng Adaptive Binarize) để chữ đanh đen, nền vùng chữ trắng xóa.
   - **Với vùng PICTURE (Hình màu):** Cắt riêng bức ảnh ra. Dừng không chạy Binarize hay Shadow Removal. Chỉ chạy module **Descreening (Lọc Moire bằng AI)** và làm sắc nét (Sharpen). Giữ nguyên dải màu RGB gốc.

3. **Gộp lớp (Multi-layer Composition):**
   - Tổng hợp lại: Đặt vùng hình màu (RGB) ốp lên trên lớp phông nền văn bản (Đen trắng). Lúc này ta sẽ có 1 trang PDF có nền trắng tinh, chữ cực nét, và hình minh họa thì lên màu sống động rực rỡ y như bản in thực tế.
