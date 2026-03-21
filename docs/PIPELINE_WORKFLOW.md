# Document Scanner — Pipeline Workflow (Cập nhật theo thực tế code Pipeline With ML)

> 🟢 = Image Processing &nbsp;|&nbsp; 🔴 = Machine Learning

---

## STEP 1: PHÁT HIỆN VÙNG TÀI LIỆU (Document Detection)

*Lưu ý: Khác với phương pháp xử lý ảnh (Image Processing) truyền thống dựa vào Canny và approxPolyDP, Pipeline ML thiết kế để đâm thẳng vào các mô hình học máy chuyên dụng nhằm đạt được độ chính xác tuyệt đối mà không cần qua Falling-back truyền thống.*

**Luồng A: Nhổ nền triệt để bằng U²-Net (Rembg - Mặc định cho cờ `--u2net`) 🔴 ML**
- **Làm gì:** Dùng mạng nơ-ron học sâu U²-Net đục thủng background 100%, bảo vệ nguyên vẹn ngay cả một tờ giấy bị nhàu nát lượn sóng, không bóp méo hay ép thành 4 góc cứng nhắc. Bức ảnh được cắt và đóng lên tấm nền trắng tinh mới.
- **Output:** Tờ giấy cong lượn tự nhiên trên nền trắng.

**Luồng B: Trích xuất góc ML Segmentation (DocAligner / YOLOv8 - Chế độ lai) 🔴 ML**
- 1a. *Tiền xử lý 🟢*: Thu nhỏ ảnh, mờ Gaussian để dễ phân tích bề mặt.
- 1b. *Phân đoạn AI 🔴*: Thay vì dùng Hough Lines thủ công, hệ thống dùng **DocAligner** chuyên dụng để nội suy 4 viền cực khít. Nếu máy không cài DocAligner, mã nguồn lùi về **YOLOv8 Segmentation** bao quanh vùng mặt nạ giấy, từ đó nội suy lấn vẽ đa giác thu gom 4 tọa độ góc.
- **Output:** Mảng 4 tọa độ góc trên khung ảnh hiện tại, kèm theo Mask nhị phân.

---

## STEP 2: BIẾN ĐỔI HÌNH HỌC VÀ LÀM PHẲNG (Geometric & Dewarping)

### 2a. Perspective Transform (Biến đổi phối cảnh vuông góc) 🟢 Image Processing
- **Khi nào chạy:** Khi bước 1 chạy bằng Luồng B (Trích xuất 4 góc của YOLO/DocAligner).
- **Làm gì:** Tính ra ma trận 3x3 và dùng `warpPerspective` để kéo màn hình chéo lật nghiêng thành khung chữ nhật chuẩn hóa chính diện.
- **Lưu ý:** Nếu hệ thống chạy bằng Luồng A (bóc mặt nạ U²-Net), bước biến đổi Phối cảnh này sẽ cố ý bị loại bỏ để bảo toàn dốc độ cong lượn vật lý nguyên bản cho quá trình ủi dòng ở 2b.
- **Output:** Ảnh tài liệu duỗi màn hình ngang dọc nhìn đối xứng tuyến tính.

### 2b. Text-line Dewarping (Phân tích nén phẳng trục dòng chữ) 🔴 Machine Learning
- **Khi nào chạy:** Tự động kích hoạt nối gót 2a hoặc 1a nếu có sự hiện diện của thư viện `page-dewarp`.
- **Làm gì:** Mô hình AI phân tích độ võng/bẻ lượn cong vút của từng hàng chữ bên trong trang sách cuốn mép. Từ đó, xây dựng một vòm lưới Spline lặn ngược để nắn/bẻ đảo chiều uốn cong của từng pixel mảnh giấy.
- **Output:** Tờ giấy phẳng lỳ như vừa được kẹp bàn ủi nhiệt độ cao. (Nếu model không gắn được file weights, hoặc lỗi phân tích, kết quả tự động rơi thẳng về mốc ảnh 2a).

---

## STEP 3: TĂNG CƯỜNG CHẤT LƯỢNG (Image Enhancement)

### 3a. Khôi phục lóa sáng Flash (Glare Removal/In-painting) 🟢 Image Processing
- **Làm gì:** Chuyển xám và phân ngưỡng mức cực cao (>250) để tách vùng đốm trắng lóa do Flash điện thoại. Nếu đốm sáng nhỏ (< 5% trang giấy), thuật toán `cv2.inpaint` sẽ nội suy vá tự động phục dựng vùng giấy bị mù chữ dựa vào pixel lân cận.
- **Output:** Loại bỏ đốm lóa chói sáng trên mặt giấy bóng cứng.

### 3b. Chống Rung Nhoè nét chữ (Unsharp Masking/Deblurring) 🟢 Image Processing
- **Làm gì:** Sửa lỗi Motion Blur do người dùng chụp bị rung tay vòng lấy nét (AF) lỏng. Trừ ảnh hiện tại cho bản nhòe Gaussian (`addWeighted` kéo biên độ) nhằm khuếch đại nếp gấp viền.
- **Output:** Kéo viền mép chữ trở nên sắc như dao cạo, khôi phục độ rõ cho sợi mực.

### 3c. Khử bóng loang lổ (Division-based Shadow Normalization) 🟢 Image Processing
- **Làm gì:** Áp dụng mảng Division-based Illumination. Dùng `MORPH_CLOSE` kernel lớn (21x21) ăn mòn mất gốc chữ đen, chỉ giữ lại độ râm tạo thành "Bản đồ phông nền" khuếch tán. Đem từng Pixel thực tế chia cho Bản đồ nền này (x255) khiến các mảng bóng tối tự dội ngược tỷ lệ sáng lên đồng đều với toàn trang.
- **Output:** Bức ảnh xám với ánh sáng tờ giấy dàn phân bổ hoàn hảo không gợn bóng tay.

### 3d. Binarization Toàn cục (Global Otsu Thresholding) 🟢 Image Processing
- **Làm gì:** Vì bước 3c đã giải quyết triệt để sự cố bóng hắt (Shadow), thao tác chốt hạ sử dụng phương pháp **Global Otsu Thresholding** (Thay vì phương pháp khoanh cửa sổ Adaptive quá gắt làm bào mòn và bẻ gãy nét mỏng). Otsu giáng 1 đường chẻ đôi rành mạch các Pixel Mực In xuống Trắng/Đen tuyệt đối dựa vào chính histogram chuẩn.
- **Output:** ✅ **Ảnh tài liệu scan (Mực đen, Giấy trắng) rành mạch nguyên gốc, chữ không gãy nát, không vỡ mảnh**.

---

## Tổng kết cơ chế tích hợp ML

Bằng việc lồng Mạng Nơ-ron (Rembg/U2Net/DocAligner/YOLO) vào **Bước 1**, và Trí tuệ AI vuốt nếp dòng chữ vào **Bước 2**, Pipeline xử lý ảnh truyền thống trước đây đã được lột xác trở thành 1 hệ thống **Front-line Machine Learning chuyên sâu**. Thay vì mò mẫm vẽ đường bờ bao bằng xử lý điểm ảnh OpenCV dễ trật nhịp, Machine Learning đánh chặn ngay vào não điểm ảnh, đẩy OpenCV (Bước 3) về vị trí Tăng Cường Sinh Học đầu cuối mang lại một hệ thống Scanner App chuẩn hóa quốc tế mạnh mẽ.
