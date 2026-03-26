# Document Scanner — Pipeline Workflow (Cập nhật theo thực tế code Pipeline With ML)

> 🟢 = Image Processing &nbsp;|&nbsp; 🔴 = Machine Learning

---

## STEP 1: PHÁT HIỆN VÙNG TÀI LIỆU (Document Detection)

*Lưu ý: Khác với phương pháp xử lý ảnh (Image Processing) truyền thống dựa vào Canny và approxPolyDP, Pipeline ML thiết kế để đâm thẳng vào các mô hình học máy chuyên dụng nhằm đạt được độ chính xác tuyệt đối mà không cần qua Falling-back truyền thống.*

**Luồng A: Nhổ nền triệt để bằng U²-Net (Rembg - Mặc định cho cờ `--u2net`) 🔴 ML**
- **Làm gì:** Dùng mạng nơ-ron học sâu U²-Net đục thủng background 100%, bảo vệ nguyên vẹn ngay cả một tờ giấy bị nhàu nát lượn sóng, không bóp méo hay ép thành 4 góc cứng nhắc. Bức ảnh được cắt và đóng lên tấm nền trắng tinh mới.
- **Output:** Tờ giấy cong lượn tự nhiên trên nền trắng.

**Luồng B: Trích xuất góc ML Segmentation (DocAligner) 🔴 ML**
- 1a. *Tiền xử lý 🟢*: Thu nhỏ ảnh, mờ Gaussian để dễ phân tích bề mặt.
- 1b. *Phân đoạn AI 🔴*: Thay vì dùng Hough Lines thủ công, hệ thống dùng **DocAligner** chuyên dụng để tìm góc. Mã nguồn sinh mặt nạ giấy, từ đó sử dụng Bounding Box xoay (`minAreaRect`) để bao bọc và thu gom 4 tọa độ góc.
- **Output:** Mảng 4 tọa độ góc trên khung ảnh hiện tại, kèm theo Mask nhị phân.

---

## STEP 2: BIẾN ĐỔI HÌNH HỌC VÀ LÀM PHẲNG (Geometric & Dewarping)

### 2a. Perspective Transform (Biến đổi phối cảnh vuông góc) 🟢 Image Processing
- **Khi nào chạy:** Khi bước 1 chạy bằng Luồng B (Trích xuất 4 góc của DocAligner).
- **Làm gì:** Tính ra ma trận 3x3 và dùng `warpPerspective` để kéo màn hình chéo lật nghiêng thành khung chữ nhật chuẩn hóa chính diện.
- **Lưu ý:** Nếu hệ thống chạy bằng Luồng A (bóc mặt nạ U²-Net), bước biến đổi Phối cảnh này sẽ cố ý bị loại bỏ để bảo toàn dốc độ cong lượn vật lý nguyên bản cho quá trình ủi dòng ở 2b.
- **Output:** Ảnh tài liệu duỗi màn hình ngang dọc nhìn đối xứng tuyến tính.

### 2b. Text-line Dewarping (Phân tích nén phẳng trục dòng chữ) 🔴 Machine Learning
- **Khi nào chạy:** Tự động kích hoạt nối gót 2a hoặc 1a nếu có sự hiện diện của thư viện `page-dewarp` (cờ `--dewarp-ml`).
- **Làm gì:** Mô hình AI phân tích độ võng/bẻ lượn cong vút của từng hàng chữ bên trong trang sách cuốn mép. Từ đó, xây dựng một vòm lưới Spline lặn ngược để nắn/bẻ đảo chiều uốn cong của từng pixel mảnh giấy.
- **Output:** Tờ giấy phẳng lỳ như vừa được kẹp bàn ủi nhiệt độ cao. (Nếu model không gắn được file weights, hoặc lỗi phân tích, kết quả tự động rơi thẳng về mốc ảnh 2a).

### 2c. Neural Grid-based Document Unwarping (UVDoc) 🔴 Machine Learning
- **Khi nào chạy:** Khi người dùng truyền cờ `--uvdoc` (thường kết hợp với `--u2net`).
- **Làm gì:** **UVDoc** sử dụng Mạng Nơ-ron Đa lớp (ResNet) phân tích tài liệu để dự đoán ra một lưới tọa độ điểm 2D/3D (Neural Grid) biểu diễn độ nhăn nheo, cong vênh sọc dưa sâu sắc trên toàn bộ diện tích giấy.
- **⚡ MỚI: Tính năng Giám sát phẳng IoU (Anti-Pinch):** Vì UVDoc sẽ làm bóp méo 4 góc vuông của 1 tờ giấy phẳng (Pinch Effect), hệ thống tự động sinh ra một chốt chặn Toán Học:
  - Khởi tạo 1 Đa giác 4 góc đường thẳng (Ideal Polygon) bao trùm tài liệu.
  - Tính Toán Diện Tích Tương Giao (Intersection over Union - IoU) giữa viền thực tế của U2-Net và Đa giác lý tưởng này.
  - Nếu `IoU > 94%`: Tờ giấy được kết luận là "Hình Phẳng Hoàn Toàn". Thuật toán tự động tước quyền chạy UVDoc và ném về **2a (Perspective Warp)** để 4 góc được vuốt thẳng tắp như dao lam.
  - Nếu `IoU < 94%`: Tờ giấy dấn mép, lõm gáy sách. Thuật toán giữ quyền cho phép UVDoc bóc tách lưới 3D Neural.
- **Output:** Khôi phục độ nén cong của vật thể vật lý mà không làm rách viền. Độ vẹn toàn cực cao.

---

## STEP 3: TĂNG CƯỜNG CHẤT LƯỢNG (Image Enhancement)

### 3a. Khôi phục lóa sáng Flash (Glare Removal/In-painting) 🟢 Image Processing
- **Làm gì:** Chuyển xám và phân ngưỡng mức cực cao (>250) để tách vùng đốm trắng lóa do Flash điện thoại. Nếu đốm sáng nhỏ (< 5% trang giấy), thuật toán `cv2.inpaint` sẽ nội suy vá tự động phục dựng vùng giấy bị mù chữ dựa vào pixel lân cận.
- **Output:** Loại bỏ đốm lóa chói sáng trên mặt giấy bóng cứng.

### 3b. Chống Rung Nhoè nét chữ (Unsharp Masking/Deblurring) 🟢 Image Processing
- **Làm gì:** Sửa lỗi Motion Blur do người dùng chụp bị rung tay vòng lấy nét (AF) lỏng. Trừ ảnh hiện tại cho bản nhòe Gaussian (`addWeighted` kéo biên độ) nhằm khuếch đại nếp gấp viền.
- **Output:** Kéo viền mép chữ trở nên sắc như dao cạo, khôi phục độ rõ cho sợi mực.

### 3c. Khử bóng tự nhiên duy trì Màu thực (RGB Independent Shadow Normalization) 🟢 Image Processing
- **Làm gì:** Trong khi các phương pháp Scanner khác ước lượng nền bằng màu xám, dẫn đến việc Bóng màu (bóng đổ của ánh sáng vàng/xanh môi trường) bị chia sai tỷ lệ gây ra Vết Ám Tím/Xám đậm xung quanh nét chữ. Hệ thống áp dụng một Cơ chế hoàn thiện cực cao:
  - **Tách 3 Kênh BGR Độc lập:** Mỗi kênh (Xanh, Lục, Đỏ) được lấy mẫu phông nền một cách tách biệt.
  - **Ghost Scale + Dilate:** Thu nhỏ ảnh cực nhanh, dùng `cv2.dilate` (Phình to) nuốt chửng toàn bộ chi tiết thẫm (mực).
  - **MedianBlur + GaussianBlur (Radius 51x51):** Biến mảng Pixel phông nền thành màng Lưới Điện Toán Ánh Sáng mượt mà. Phóng to trả lại nguyên hiện trạng.
  - Lấy từng pixel tài liệu gốc chia thẳng cho Bản đồ Phông Nền tương ứng của riêng kênh đó (`Thuyết Nguồn Sáng Kép`).
- **Output:** Cân bằng lại cả Độ Sáng lẫn **Nhiệt Độ Màu** của vùng khuất bóng. Kết quả là một bức ảnh trắng phau nhưng duy trì nguyên vẹn màu Mực Xanh, Dấu Mộc Đỏ không dính một tì vết ám xám tàng hình nào.

### 3d. Phơi Sáng Thích Ứng (Adaptive Histogram Stretching & CLAHE) 🟢 Image Processing
- **Làm gì:** Dựa trên phân tích nhược điểm của Cắt Ngưỡng cố định (Fixed Binarize Thresholds gây cháy nếu sáng chênh lệch), Pipeline tích hợp hoàn toàn cơ chế tự thích nghi tương phản (Adaptive) 2 Chế Độ:
  - **Tương phản cục bộ:** Dùng Mạng Phân Vùng Lưới `CLAHE` quét 8x8 pixels để đẩy bật các vùng chữ nét bút chì chìm trong lều sấp bóng râm.
  - **Tự tính ngưỡng bách phân vị:** Kéo Histogram thành mảng 1D, lấy mốc Percentile để trích xuất linh động `Black Point` (ngưỡng triệt chữ đen $2\%$) và `White Point` (vùng nền trắng sáng cháy $95\%$).
  - **Mượt biên tự dãn (Anti-Aliasing):** Khoảng màu giao thoa được duỗi mềm tuyến tính, bảo vệ hình thái đường cong của nét tròn mà không chém răng cưa.
  - **Lọc Khử Nhiễu:** Trang màu được trích xuất `LAB Denoising` chỉ đánh gắt trên $A,B$ để tấy vết mực xám bẩn mà không hư font chữ; trang Đơn Màu được quét phẳng nền bằng `Bilateral Filter`.
- **Output:** ✅ **Ảnh tài liệu đạt chất lượng scanner chuẩn Enterprise (Nền trắng đồng đều vô cực 100%, 100% không còn dính rác nhiễu Signal Noise do khuếch đại ánh sáng)**.

---

## Tổng kết cơ chế tích hợp ML

Bằng việc lồng Mạng Nơ-ron (Rembg/U2Net/DocAligner) vào **Bước 1**, và Trí tuệ AI vuốt nếp dòng chữ vào **Bước 2**, Pipeline xử lý ảnh truyền thống trước đây đã được lột xác trở thành 1 hệ thống **Front-line Machine Learning chuyên sâu**. Thay vì mò mẫm vẽ đường bờ bao bằng xử lý điểm ảnh OpenCV dễ trật nhịp, Machine Learning đánh chặn ngay vào não điểm ảnh, đẩy OpenCV (Bước 3) về vị trí Tăng Cường Sinh Học đầu cuối mang lại một hệ thống Scanner App chuẩn hóa quốc tế mạnh mẽ.
