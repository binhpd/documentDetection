# Giải Thích Chi Tiết: STEP 1 - PHÁT HIỆN VÙNG TÀI LIỆU (Document Detection)

Mục tiêu của Bước 1 là "tìm thấy" tờ giấy nằm ở đâu trong khung hình không gian bức ảnh, sau đó tạo ra một lớp mặt nạ (Mask) để niêm phong khu vực đó lại và trích xuất tọa độ 4 góc của tờ giấy.

---

## 1. Thuật toán Tiền Nhận Diện (Preprocessing)
Trước khi đẩy hình ảnh vào các mạng lưới trí tuệ nhân tạo, hình ảnh phải được chuẩn hóa để tiết kiệm bộ nhớ và triệt tiêu bớt đặc điểm gây nhiễu.

### 1.1 Khử nhiễu làm mờ (Gaussian Blur)
* **Ý nghĩa học thuật:** Lọc bỏ tín hiệu nhiễu tần số cao (High-frequency noise).
* **Thuật toán sử dụng:** Bộ lọc không gian Gaussian (`cv2.GaussianBlur`). Thuật toán sử dụng một ma trận nhân chập (Convolution Kernel), thường là kích thước $5\times5$ hoặc $9\times9$. Tính toán giá trị pixel trung tâm dựa trên phân phối hình chuông (Normal Distribution) của các pixel lân cận, giúp làm mờ bề mặt thô nhám của bàn gỗ, sàn xi măng nhưng vẫn giữ được hình hài tờ giấy.
* **Input - Output:** Nhiễu hạt / Vân thảm cỏ $\rightarrow$ Nề nếp mịn màng.

### 1.2 Hạ độ phân giải (Downscaling)
* **Ý nghĩa học thuật:** Các bức ảnh chụp từ smartphone đời mới thường là $3000 \times 4000$ pixel (12MP). Nếu đưa thẳng vào AI sẽ gây sập RAM và tốc độ cực độ chậm.
* **Thuật toán sử dụng:** Ánh xạ tỷ lệ đa tuyến tính (Bilinear Interpolation) qua `cv2.resize`. Ảnh được scale xuống kích thước thu nhỏ (mặc định chiều dài ~500px).
* **Ratio Binding:** Biến số `ratio` (Tỷ lệ = Ảnh Gốc / Ảnh Thu Hẹp) được lưu lại. Về sau mọi tọa độ dò tìm được trên ảnh thu nhỏ sẽ được nhân trỏ với `ratio` để bung ngược chính xác về bức ảnh gốc độ phân giải cao.

---

## 2. Thuật toán Trích xuất Vùng Giấy (Semantic Segmentation)
Bước này dùng "Trí thông minh" để phát quang bụi rậm. Ta có 2 luồng thuật toán chạy song song tùy cấu hình khởi tạo.

### Luồng A: Salient Object Detection (U²-Net)
* **Ý nghĩa học thuật:** Bóc vật thể nổi bật nhất trước ống kính (Foreground Extraction).
* **Tại sao lại dùng?** Xử lý ảnh tĩnh kinh điển thường đi tìm đường kẻ (Lines) dựa trên Canny. Việc dò đường kẻ bằng Canny rất dễ bị đứt đoạn do rách giấy gập mép hoặc nét thảm gạch ngói sọc dưa. U²-Net không quét đường nét, nó đánh giá Ngữ Nghĩa của từng cục Pixel.
* **Input:** Ảnh thu gọn (RGB).
* **Xử lý:** Mạng U²-Net quét qua ảnh ở nhiều tầng cấu trúc, nén và bung các đặc trưng thị giác chéo để khoanh ra toàn bộ đám pixel thuộc về "Tờ giấy".
* **Output:** Mặt nạ Rỗng (Alpha Channel). Giấy là mảng Trắng (255), Background là Đen (0).

### Luồng B: Image Segmentation (DocAligner)
* **Ý nghĩa học thuật:** Dò tìm thực thể riêng biệt.
* **Xử lý:** DocAligner phát hiện lớp đối tượng "Tài liệu" và tô đè một lớp mặt nạ (Mask) phân đoạn theo lề cạnh của nó.

---

## 3. Thuật toán Hình Học Rút Góc (Bounding Box Approximation)
Có được đám mây màu trắng (Mask) rồi, làm sao để rút ra được 4 con số Tọa độ $[X, Y]$ cho các góc tờ giấy?

### 3.1 Dò quỹ đạo đường bao (Contour Tracing)
* **Thuật toán sử dụng:** Suzuki85 algorithm (`cv2.findContours`).
* **Input:** Mask nhị phân (Đen Trắng) từ thư viện phần 2.
* **Xử lý:** Trực tiếp nương theo ranh giới vùng màu Trắng - Đen, liên kết các pixel ngoài viền lại với nhau thành 1 vòng khép kín.

### 3.2 Khoanh Hình chữ nhật nhỏ nhất (Minimum Area Bounding Box)
* **Thuật toán sử dụng:** Thuật toán Kẹp Hình `cv2.minAreaRect(contour)` kết hợp Khai thác điểm `cv2.boxPoints()`.
* **Cơ chế:**
  1. Xây ra một hình hộp (Bounding box) ôm lấy vòng biên giấy (`contour`).
  2. Bắt đầu quay (Rotate) hình hộp đó xung quanh trọng tâm với nhiều góc độ khác nhau.
  3. Ở mỗi góc xoay, ép hình hộp lại cho đến khi nó cấn/dính cả 4 cạnh vào khối Mask của tờ giấy.
  4. Chọn ra cái Khung hình chữ nhật có diện tích chật dính nhất ($W \times H$ nhỏ nhất).
* **Output:** 4 cặp tọa độ $(X,Y)$ biểu diễn cho 4 mốc cực của hình hộp đó. Tọa độ này chốt sổ Bước 1, được đẩy sang Bước 2 để nắn mảnh giấy đập vào khuôn hình hộp này.
