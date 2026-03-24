# KHUNG SLIDE BÁO CÁO DỰ ÁN DOCUMENT RESTORATION (PIPELINE)

## SLIDE 1: TRANG BÌA (Title Slide)
- **Tên đề tài:** Giải pháp khôi phục và tăng cường chất lượng hình ảnh tài liệu quét từ camera thiết bị di động (Document Restoration & Enhancement).
- **Môn học:** (Ghi tên môn học)
- **Nhóm thực hiện:** Nhóm 6
- **Giáo viên hướng dẫn:** (Tên GV)

## SLIDE 2: NỘI DUNG BÁO CÁO (Agenda)
1. Đặt vấn đề: Những thách thức khi chụp ảnh tài liệu bề mặt.
2. Tổng quan nghiên cứu & Các bài toán liên quan.
3. Giải pháp tăng cường chất lượng và tương phản hiện hành.
4. Phạm vi dự án (Scope) & Phương pháp tiếp cận (Pipeline đề xuất).
5. Kết quả thực nghiệm và đánh giá.

---

## PHẦN 1: ĐẶT VẤN ĐỀ

### SLIDE 3: Đặt vấn đề - 20 Thách thức vật lý khi chụp ảnh tài liệu bằng Smartphone
*So với máy quét Scanner chuyên dụng bị cố định điều kiện môi trường, thiết bị di động tự do phải đối diện với vô vàn lỗi từ phổ thông đến cực đoan, chia thành 4 nhóm chính:*

**Nhóm 1: Giao thoa Ánh sáng rắc rối (Illumination & Lighting Issues)**
1. Đổ bóng bàn tay/ngón tay người chụp lên văn bản.
2. Bóng đen nguyên khối của thiết bị điện thoại in lên tâm mặt giấy.
3. Độ rọi tản sáng (Gradient) không đều (Nửa tờ giấy nằm cạnh cửa sổ lóa sáng, nửa nằm trong bóng râm).
4. Ánh sáng đèn huỳnh quang đập lóa thành dải dài dọc theo biên giấy bóng.
5. Cực đoan: Bật đèn chiếu Flash tạo thành đốm mù trắng xóa (Glare) che khuất hoàn toàn chữ trên bề mặt.
6. Môi trường cực kỳ thiếu sáng (Low-light) sinh ra hiện tượng nhiễu hạt bụi (ISO Noise/Salt-Pepper noise).
7. Sai lệch phổ màu trắng do môi trường ám ánh đèn đỏ/vàng, dẫn đến thuật toán khó phân biệt nền giấy và mực in khác màu.

**Nhóm 2: Biến dạng Không gian Hình học (Geometric & Spatial Distortions)**
8. Lệch góc nhìn phối cảnh (Perspective Distortion) do tư thế cầm chéo điện thoại (dẫn tới màng giấy hình thang lệch).
9. Rìa góc giấy in nếp gập gẫy, bị quăn mép hoặc vạt gió bay lên.
10. Biến dạng uốn lượn phi tuyến tính (Curved lines) do nếp cuốn vồng lên của vùng gáy các cuốn sách quá dày.
11. Bề mặt nhấp nhô lồi lõm trầm trọng do tờ giấy bị vò nát/nhàu nhĩ rồi vuốt phanh ra.
12. Quang sai vật lý do cụm thấu kính góc rộng của smartphone làm tờ giấy rảo phình ra ở giữa và bóp méo ở rìa.

**Nhóm 3: Sai hỏng Tiêu cự & Rung máy (Camera Hardware Constraints)**
13. Rung mờ chuyển động (Motion Blur) do thao tác bấm nút chụp khiến tay người rung lắc.
14. Nhòe do mất tâm lưới lấy nét tự động (Out-of-focus Blur) khiến chữ viền bị nhòe tịt không còn nét ranh giới.
15. Chụp ở khoảng cách quá xa (hụt Crop) dẫn tới hạt Pixel phân giải cực thấp không đủ biểu diễn độ cong của 1 ký tự.
16. Các dòng kẻ lưới nền trên giấy bị dính nhiễu răng cưa (Aliasing effect) khi độ chi tiết máy ảnh quá thấp (Moire pattern).

**Nhóm 4: Sự xuống cấp nội tạng của Tài liệu (Document Surface Degradations)**
17. Nền giấy cũ mốc ngả vàng lốm đốm ố, hoặc dính vết bẩn/cặn cà phê trên bề mặt.
18. Thấm, hằn rãnh mực mặt sau (Bleed-through): Các nét bút viết đậm đè từ trang sau lưng thấm xuyên mờ ảo lên nền mặt trước giấy mỏng.
19. Mực phai trôi theo hệ số thời gian (Faded ink), nét chữ tẻ nhạt, mất vạch đứt đoạn không liền mạch.
20. Cực đoan: Các văn bản hành chính với vô vàn chữ viết tay nghuệch ngoạc chồng chéo lấn đè lên viền bảng, và bị các con dấu mộc đỏ in đè chèn lấp che lên mặt chữ đen.

---

## PHẦN 2: TỔNG QUAN HƯỚNG NGHIÊN CỨU HIỆN TẠI

### SLIDE 4: Các bài toán nghiên cứu liên quan đến Document Restoration
*Nêu các hướng nghiên cứu hiện nay trên thế giới để giải quyết nhóm vấn đề trên:*
- **Nghiên cứu về Trích xuất và Nắn chỉnh phối cảnh (Document Localization):** Từ các phương pháp cổ điển dò cạnh (Edge/Contour Detection như Hough, Canny) đến hiện đại hóa bằng Deep Learning (Segment ranh giới bằng U-Net, YOLO).
- **Nghiên cứu về Vuốt thẳng dòng chữ cong (Dewarping):** Dùng lưới tọa độ Spline nội suy 3D để uốn lại mặt phẳng cong (ví dụ thuật toán text-line dewarping).
- **Nghiên cứu về Tăng cường (Enhancement):** Làm sạch rác tín hiệu, Binarization (nhị phân hóa) đưa ảnh màu sặc sỡ về bản trắng đen kỹ thuật số tinh giản.

### SLIDE 5: Giới hạn của các giải pháp Binarize & Tăng cường truyền thống
*Phân tích các thuật toán Binarization kinh điển thường dùng để kéo tương phản, và chỉ ra điểm yếu (Làm cớ để bài mình đưa ra Soft Binarize):*
- **Global Thresholding (Otsu):** Áp một ngưỡng cho toàn ảnh. Do ảnh chụp bằng điện thoại thường bị bóng đổ (Shadow), vùng bóng bị Otsu nhuộm đen sì toàn bộ thay vì cắt được chữ.
- **Adaptive Thresholding (Phân ngưỡng cục bộ):** Giải quyết được bóng gắt, nhưng cửa sổ nội suy quá khắt khe khiến nét chữ thanh mảnh lờ mờ bị cắt đứt gãy, gây đứt đoạn ký tự (Vấn đề "xóa thấm" đục thủng nét chữ). 

---

## PHẦN 3: ĐỀ XUẤT GIẢI PHÁP CỦA DỰ ÁN

### SLIDE 6: Xác định Phạm Vi Dự Án (Project Scope)
Dựa trên các bài toán nan giải trên, dự án tập trung vào:
- **Đầu vào:** Bức ảnh chụp tài liệu thô trong điều kiện yếu (cong, nghiêng, lóa, bóng dâm).
- **Mục tiêu thao tác:** Tạo ra một **Hybrid Pipeline liên hoàn** ưu việt, kết hợp giữa Sức mạnh đục nền của Machine Learning và độ lách viền tơ tóc của Image Processing OpenCV.
- **Đầu ra kỳ vọng:** Một trang tĩnh "ảo" vuông vức 100%, nét mực đen giữ được độ êm mượt tự nhiên, nền trắng vô cực không răng cưa bóng mờ.

### SLIDE 7: Phương Pháp Tiếp Cận (The Core Pipeline)
*Sơ đồ luồng 3 bước (Vẽ sơ đồ luồng):*
- **STEP 1 [ML Detection]:** Bứt phá khỏi Canny truyền thống. Dùng AI (DocAligner/YOLO/U2Net) để ép khung và bóc hẳn cả cấu trúc tờ giấy lách qua phông nền tạp nham.
- **STEP 2 [Geometric Dewarping]:** Can thiệp tọa độ — Chiếu vuông góc (Perspective Transform) và Nắn đường lượn sóng (Text-line Dewarping).
- **STEP 3 [CV Enhancement]:** Tăng cường chốt chặn 4 khâu hoàn chỉnh: Xóa đốm Lóa -> Kéo gắt viền chữ rung nhòe -> Triệt tiêu bóng loang -> **Phơi sáng Xám Tuyến Tính (Soft Binarize)**.

### SLIDE 8: Điểm sáng Công nghệ: Soft Binarization (Linear Contrast Stretch)
*Giải thích rõ tại sao không dùng Otsu/Adaptive.*
- Thay vì chém đứt pixel theo chuẩn vỡ hạt 0-255 của Binarize đời cũ làm gai mảnh vỡ chữ, chúng ta áp dụng Piecewise Linear Stretching.
- Khóa chặt Điểm Trắng (White Point) để đẩy bóng mù và lấm tấm rác lên Trắng xóa 255. Khóa chặt Điểm Đen (Black Point) lôi mực phai chìm xuống lõi 0.
- Giữ khu vực trung gian đóng vai trò Anti-Aliasing, bảo vệ được sự quyện xám ở viền biên chữ, giúp chữ **mềm mại tuyệt đối** khi xuất PDF.

---

## PHẦN 4: KẾT QUẢ TRIỂN KHAI THỰC NGHIỆM (Implementation & Results)

### SLIDE 9: Thực nghiệm Step 1 & 2 (Cắt và Ép mặt phẳng)
*Chèn ảnh so sánh (Before / After):*
- Hình 1: Ảnh gốc trên bàn bừa bộn + chụp nghiêng chéo.
- Hình 2: Kết quả bóc nền (Ảnh mask đỏ) bởi YOLO / DocAligner / Rembg.
- Hình 3: Kết quả sau biến đổi ma trận — Tờ giấy đã được lật đứng chóp 1 cách ngay ngắn và vuốt nếp rãnh gáy sách.

### SLIDE 10: Thực nghiệm Step 3.1 & 3.2 (Khử Lóa & Khử Rung)
*Chèn ảnh soi zoom sát chữ:*
- Vết lốm đốm của đèn flash điện thoại đã được trám lại (Inpainting).
- Các viền chữ mờ nhòe (Motion blurs) được dao cạo sắc gắt lại bằng Unsharp Masking.

### SLIDE 11: Thực nghiệm Step 3.3 (Khử Bóng Loang Lổ Khó Chịu)
*Chèn ảnh trước/sau khi qua hàm Division-based Shadow Removal:*
- Mặt giấy bóng đen bàn tay thui thủi được cân bằng (Illumination Normalization). Phông nền trở về xám bạc sáng đồng đều.

### SLIDE 12: Đỉnh cao Binarization (So sánh Options)
*Trình chiếu cắt lát 2 hoặc 3 options khác nhau của thuật toán Phơi sáng mềm Soft Binarize:*
- Nhấn mạnh hình thái chữ cắt Binarize thông thường (gai góc, lỗ rỗ) bên cạnh con chữ của Option hiện tại (nền đanh trắng phau, nhân đen thủng, nhưng vành cong con chữ có độ sương muối khử răng cưa mượt mắt).

---

## SLIDE 13: KẾT LUẬN & HƯỚNG MỞ RỘNG
**1. Kết luận:**
- Pipeline mang tính thực tiễn vô cùng cao.
- Giải quyết ráo riết khuyết điểm bẻ chữ của phương pháp Adaptive Window Scanner xưa cũ.
**2. Hướng mở rộng:**
- Đưa trọng lượng Neural Network lên các bản Mini (Edge Devices) để chạy trực tiếp Pipeline này trên thiết bị Mobile tiết kiệm Ram/Pin.

---
*(Trang Cảm Ơn / Q&A)*
