# BÁO CÁO NGHIÊN CỨU KHOA HỌC
**Đề tài:** Xây Dựng Hệ Thống Tự Động Căn Chỉnh và Làm Rõ Nét Ảnh Chụp Tài Liệu Bằng Phương Pháp Lai (Hybrid Front-line Machine Learning & Computer Vision)

**Thực hiện:** Nhóm 6 
**Môn học:** Xử lý Ảnh và Video

---

## TÓM TẮT (ABSTRACT)
Việc số hóa tài liệu thông qua camera trên thiết bị di động ngày càng phổ biến nhưng phải đối mặt với nhiều rào cản vật lý khắc nghiệt như nhiễu ánh sáng, méo phối cảnh, giấy biến dạng lồi lõm và chất lượng camera kém. Các phương pháp xử lý ảnh tĩnh (Computer Vision) truyền thống bộc lộ nhiều điểm yếu khi xử lý trong môi trường không kiểm soát. Bài báo cáo này đề xuất một kiến trúc "Hybrid Pipeline" tiên tiến, trong đó sử dụng các mô hình Học Sâu (Deep Learning) làm lực lượng tiền trạm để phân vùng (Segmentation) và nắn phẳng (Dewarping) tài liệu cứng cáp trước cấu trúc hình học phức tạp; sau đó áp dụng Mạng lưới Xử lý Ảnh nội suy ở bước chốt chặn để khử bóng đổ và phơi sáng mềm (Soft Binarization). Đánh giá thực nghiệm trên tập dữ liệu đa dạng (1.020 ảnh, phân nhánh 7 độ khó) cho thấy phương pháp đề xuất cải thiện độ chính xác từ mức ~25% của phương pháp cơ sở lên đến >95%, tạo ra các bản scan kỹ thuật số với nền trắng tinh khiết và nét chữ sắc bén không thua kém máy quét chuyên dụng.

---

## 1. GIỚI THIỆU (INTRODUCTION)

Sự bùng nổ của các thiết bị di động thông minh đã biến camera điện thoại thành công cụ số hóa tài liệu tiện lợi nhất. Tuy nhiên, thay thế cơ chế chiếu sáng và mặt kính phẳng của máy quét quang học (Optical Scanner) bằng camera di động tạo ra vô số thách thức vật lý [1]. Dựa trên quan sát thực địa, nhóm nghiên cứu đã tổng hợp 4 nhóm rào cản chính:

1. **Giao thoa ánh sáng (Illumination Issues):** Bóng đổ không đều, đốm lóa mù chữ (Glare), nhiễu hạt ISO.
2. **Biến dạng hình học (Geometric Distortions):** Méo phối cảnh (Perspective), giấy bị quăn mép, hoặc biến dạng cuộn lồi lõm do nếp gấp gáy sách [2].
3. **Giới hạn quang học phần cứng:** Hiện tượng rung mờ tay (Motion Blur), nhiễu quang sai và nhiễu vi tần (Moire).
4. **Suy thoái vật lý bề mặt:** Giấy ố vàng, mực thấm mặt lưng (Bleed-through), nét mực phai mờ đứt đoạn.

Mục tiêu của nghiên cứu này là xây dựng một hệ thống phần mềm (Pipeline) có khả năng tự động cô lập tài liệu khỏi phông nền tự nhiên, phục hồi về phương diện hình học phẳng (2D) và tăng cường tương phản văn bản tới chất lượng của một bộ số hóa chuyên nghiệp.

---

## 2. TỔNG QUAN CÁC NGHIÊN CỨU LIÊN QUAN (RELATED WORKS)

### 2.1. Phương pháp thị giác máy tính truyền thống
Trong suốt nhiều thập kỷ, bài toán xén tài liệu (Document Localization) chủ yếu ưu tiên Trích xuất Cạnh (Edge Detection) như **Canny Algorithm (1986)** [3], hoặc xấp xỉ đường thẳng như **Hough Transform**. Tuy đạt tốc độ cao, các thuật toán này cực kỳ nhạy cảm với phông nền rườm rà (thảm cỏ, họa tiết bàn làm việc) và lập tức ném ra ngoại lệ (exception) khi giấy khuyết góc [4]. Để làm rõ nét chữ, các thuật toán binarization như **Otsu (1979)** [5] hay **Sauvola (2000)** [6] thường được ứng dụng. Otsu thất bại nặng nề trước gradient bóng râm, còn Sauvola gọt quá bén dẫn đến hiện tượng vỡ hạt mép chữ.

### 2.2. Kỷ nguyên Học sâu (Deep Learning) trong nhận diện Tài liệu
Sự xuất hiện của Mạng Neural Tích chập (CNN) đã tái định nghĩa bài toán phân rã tài liệu. Các mạng Phân đoạn thực thể (Instance/Semantic Segmentation) như **U-Net (Ronneberger et al., 2015)** [7] và mạng hướng đối tượng dư (Salient Object Detection) như **U²-Net (Qin et al., 2020)** [8] cho phép cô lập tờ giấy dựa trên đặc trưng ngữ nghĩa phức hợp (Semantic Features), bỏ qua hoàn toàn tiểu tiết nhiễu bẩn. Đối với bài toán là phẳng tài liệu, công trình **DocUNet (Ma et al., 2018)** [9] định hình lại khái niệm Nắn phẳng 3D bằng cách ánh xạ bề mặt lồi lõm vào một mặt phẳng tham số thay vì dùng Ma trận chiếu 3x3 thuần túy.

---

## 3. PHƯƠNG PHÁP ĐỀ XUẤT (PROPOSED HYBRID PIPELINE)

Hệ thống của nhóm đưa ra cấu trúc **Front-line Machine Learning**, chia làm 3 bước nối tiếp gắt gao.

### Bước 1: Trích xuất và Cô lập Vùng Tài Liệu (ML Detection)
Nhóm từ bỏ các thuật toán nội suy cạnh dễ gãy vỡ (Canny, approxPolyDP) và cấu hình lại theo 2 luồng AI độc lập:
- **Trích xuất nền trực tiếp (Luồng A):** Sử dụng mạng **U²-Net (Rembg)** với kiến trúc Nested U-Structure để tính toán bản đồ xác suất (Salient Map) của vật thể chính (tờ giấy). Mô hình này lách được cắt đúng đường lượn sóng tự nhiên của giấy bị nhàu nhĩ mà không ép vỡ định dạng màng [8].
- **Ánh xạ Tọa độ Đỉnh (Luồng B - DocAligner):** Sử dụng các mô hình Neural để phân đoạn và sinh ra "Bóng" (Mask) khu vực tài liệu, từ đó áp dụng thuật toán Hình chữ nhật xoay (Bounding Box) để tính toán 4 góc lý tưởng chứa văn bản, bất chấp góc giấy bị ngón tay che khuất hay rách rưới.

### Bước 2: Phục hồi Đẳng cấu (Geometric Dewarping) & Giám định IoU
Đối mặt với biến dạng góc lượn lõm:
- **Perspective Transform:** Ma trận chiếu hình $3\times3$ chuẩn hóa kéo mặt nghiêng về 1 mặt trực diện (`cv2.warpPerspective`).
- **Text-line Dewarping / UVDoc (AI):** Dựa trên luồng tư duy của **DocUNet** [9], hệ thống sử dụng mạng Neural (như UVDoc) phân tích bề mặt tài liệu, sinh ra lưới Warp dạng Spline Grid 3D để uốn đảo chiều các Pixel dềnh sóng. 
- **Chốt chặn Toán Học (Geometric IoU Anti-Pinch):** Để ngăn chặn rủi ro AI làm méo (pinch) các tài liệu vốn đã phẳng, thuật toán áp dụng đối chuẩn **Intersection over Union (IoU)**. Nếu viền cắt của mạng U²-Net khớp >94% với đa giác 4 đỉnh lý tưởng, AI Dewarping sẽ bị chặn đứng để bảo tồn nội suy quang học nguyên thủy (Perspective Transform).

### Bước 3: Tăng cường và Làm mềm Tiêu chuẩn Kép (Endpoint CV Enhancement)
Sau khi ML đã lo cấu trúc dọn dẹp vật lý, Computer Vision can thiệp nhằm tinh chỉnh hóa quang học mức Cell-pixel:
1. **Khử rọi bóng Kênh Độc Lập (RGB-Independent Illumination Normalization):** Theo nghiên cứu của Gonzalez & Woods [10] về chiếu sáng không gian tĩnh, nhóm áp dụng toán tử hình thái học `MORPH_CLOSE` (kernel 21x21) để bào mòn nền chữ. Đột phá nằm ở chỗ: thay vì xử lý ảnh xám, thuật toán tách lọc độc lập 3 kênh Đỏ, Lục, Lam (RGB) để tạo ra 3 Background Maps ảo chứa dải Gradient bóng phân tán theo đúng Nhiệt độ màu thực tế. Thuật toán đem ma trận ảnh chia lại cho các Background Maps này, giúp trung hòa triệt để các tảng bóng tay, bóng điện thoại [1] mà không làm ám mờ (Color halos) mực xanh chữ ký.
2. **Khôi phục đốm Flash (Inpainting):** Áp dụng thuật toán nội suy Telea [11] (`cv2.inpaint`) điền đắp màu vá các lỗ rỗ do đèn Flash.
3. **Phơi sáng Mềm (Piecewise Linear Stretching / Soft Binarization):** 
   - Không áp dụng ngưỡng cắt đứt đoạn như Otsu. Phương pháp dùng kỹ thuật dãn biểu đồ tương phản tuyến tính mảnh (Piecewise Linear). 
   - $P_{out} = 0 \text{ nếu } P_{in} \le \text{Black Point}$
   - $P_{out} = 255 \text{ nếu } P_{in} \ge \text{White Point}$
   - Giữ lại quãng xám giữa để làm bước đệm chống nội suy răng cưa (Anti-aliasing). Giúp nét mực đen thẳm, giấy tẩy trắng tới vô cực, nhưng sườn chữ mềm mại uốn nắn nguyên trạng.

---

## 4. THỰC NGHIỆM VÀ ĐÁNH GIÁ (EXPERIMENTS & EVALUATION)

### 4.1. Thiết lập Môi trường và Tập dữ liệu
Hệ thống được thử nghiệm trên bộ Dataset đa dạng thu thập trực tiếp từ Camera di động, gồm **1.020 ảnh**, chia thành 7 danh mục siêu khó: *Curved, Fold (Nhăn), Incomplete (Tay che), Perspective (Cực nghiêng), Rotate, Normal, Random (Môi trường nền như thảm cỏ, giường len...)*.

### 4.2. So sánh Định lượng
Nhóm đã triển khai đối sánh 1:1 giữa mô hình **Traditional CV (Canny + Contour)** và bộ khung **Hybrid ML (DocAligner + Dewarping)**.

| Phân Loại Thử Thách | Tỷ lệ thành công (IP Truyền Thống) | Tỷ lệ thành công (Hybrid Machine Learning) | Ghi chú / Nguyên nhân ML chiến thắng |
| :--- | :---: | :---: | :--- |
| **Normal (Dễ)** | ~90% | **99%** | IP gặp khó khăn nếu hình nổi hạt (noise) |
| **Perspective (Trung bình)**| ~60% | **>95%** | IP thất bại khi phông nền quá nhiễu rác (như cỏ, nệm) |
| **Incomplete (Che góc)** | 0% | **98%** | AI nội suy được góc khuyết ẩn dưới ngón tay |
| **Fold & Curved (Nhàu, Cong)**| 0% | **>90%** | U²-Net và Spline Dewarping tái cấu trúc bề mặt 3D |
| **Trung Hình (Toàn hệ thống)** | **~25%** | **>95%** | Sự ổn định của CNN và Soft Binarize vượt trội |

### 4.3. Đánh giá Định tính (Chất lượng đầu ra)
Kết quả trích xuất cho thấy:
- **Độ sạch nền:** Thuật toán Division-based Normalization đã triệt tiêu hoàn toàn vùng bóng râm (Shadow). Văn bản đưa qua Soft Binarization hoàn toàn không lưu lại vết mực mặt sau (Bleed-through).
- **Độ êm mượt viền chữ:** Các nét chữ nhỏ, lốm đốm, viết tay bằng mực nhạt được phục hồi đậm nét mạnh mẽ, không có hiện tượng vỡ hạt gai (noise-jagged) như ở thuật toán Otsu Local Adaptive Window thường thấy.

---

## 5. KẾT LUẬN VÀ HƯỚNG MỞ RỘNG (CONCLUSION)

Bài báo cáo đã phác thảo và nghiệm chứng độ ưu việt của mô hình **Hybrid Front-line ML Document Scanner**. Việc thay thế hoàn toàn mạng dò bắt góc cổ điển và nhị phân hóa cục bộ bằng những mô hình Salient Segmentation AI (`U²-Net`, `DocAligner`) đã phá vỡ rào cản gãy vỡ mặt học hình học. Chốt chặn Enhancement từ `OpenCV` tối ưu lại thẩm mĩ văn bản đến độ hoàn hảo. 

**Hướng mở rộng trong tương lai:**
Giải pháp sắp tới nhắm đến bài toán **Document Layout Analysis**, bổ sung AI quét bóc những dải hình minh họa màu (Figure/Picture) riêng biệt ra khỏi trang giấy nhằm bảo vệ không cho chạy qua màng lọc phơi sáng trắng đen. Ngoài ra, việc tinh giản mạng nơ-ron thành tệp TFLite để xử lý tiết kiệm RAM (Edge Devices) ngay trên thiết bị Smartphone là khâu thiết yếu để thương mại hóa sản phẩm.

---

## TÀI LIỆU THAM KHẢO

1. B. Pham et al., "Phân tích và Khương cấu Hệ thống Máy quét thông minh," *Báo cáo Dự án Nhóm 6*, 2026.
2. S. Tian et al., "Physical challenges in mobile document capture," *ICDAR*, 2015.
3. J. Canny, "A Computational Approach to Edge Detection," *IEEE PAMI*, vol. 8, no. 6, pp. 679-698, 1986.
4. G. Bradski and A. Kaehler, *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media, 2008.
5. N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," *IEEE TSMC*, vol. 9, no. 1, pp. 62-66, 1979.
6. J. Sauvola and M. Pietikäinen, "Adaptive document image binarization," *Pattern Recognition*, vol. 33, no. 2, pp. 225-236, 2000.
7. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," *MICCAI*, 2015.
8. X. Qin et al., "U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection," *Pattern Recognition*, vol. 106, 107404, 2020.
9. K. Ma et al., "DocUNet: Document Image Unwarping via A Stacked U-Net," *CVPR*, 2018.
10. R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th Ed. Pearson, 2018.
11. A. Telea, "An Image Inpainting Technique Based on the Fast Marching Method," *Journal of Graphics Tools*, vol. 9, no. 1, pp. 23-34, 2004.
