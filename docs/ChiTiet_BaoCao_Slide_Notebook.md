# Hướng dẫn tạo Slide từ Jupyter Notebook & Nội dung chi tiết Báo cáo

Tài liệu này cung cấp **cách cấu hình Jupyter Notebook thành Slide** cũng như **toàn bộ nội dung chi tiết và kịch bản thuyết trình (Speaker Notes)** để bạn dán trực tiếp vào các cell trong Notebook.

---

## 1. CÁCH TẠO SLIDE TRỰC TIẾP TRONG JUPYTER NOTEBOOK

Có 2 cách phổ biến nhất để biến Jupyter Notebook thành một bài báo cáo Slide chuyên nghiệp:

### Cách A (Khuyên dùng): Sử dụng Extension RISE (Live Slideshow)
RISE cho phép bạn trình chiếu Notebook của mình dưới dạng slide trực tiếp ngay trên trình duyệt mà không cần export. Bạn thậm chí có thể chạy code ML ngay khi đang thuyết trình.
1. Cài đặt RISE trong terminal Notebook của bạn:
   ```bash
   pip install RISE
   ```
2. Mở file Notebook (.ipynb) của bạn.
3. Kích hoạt chế độ gán nhãn Slide: Vào menu **View > Cell Toolbar > Slideshow**.
4. Ở mỗi Cell (Markdown hoặc Code), bạn sẽ thấy một ô chọn loại Slide (Slide Type) ở góc phải:
   - **Slide**: Cell này là một trang slide mới.
   - **Sub-Slide**: Slide phụ, nằm bên dưới slide hiện tại (nhấn phím XUỐNG để chuyển).
   - **Fragment**: Xuất hiện thêm từng đoạn trên cùng một slide (hiệu ứng xuất hiện).
   - **Notes**: Ghi chú dành riêng cho diễn giả (khán giả không thấy).
   - **Skip**: Bỏ qua cell này khi trình chiếu.
5. Nhấn nút **"Enter/Exit RISE Slideshow"** (biểu tượng đồ thị bar nhỏ trên thanh toolbar) để bắt đầu báo cáo.

### Cách B: Export sang HTML Slides (Nbconvert)
Nếu bạn muốn gửi file đi dưới dạng Web HTML tĩnh:
1. Chạy lệnh sau trong terminal để tự động gom Notebook thành 1 file HTML reveal.js:
   ```bash
   jupyter nbconvert Khung_Bao_Cao_Slide.ipynb --to slides --post serve
   ```
2. Lệnh này sẽ mở trình duyệt hiển thị luôn bản slide tĩnh cho bạn.

---
---

## 2. NỘI DUNG CHI TIẾT CÁC SLIDE & GHI CHÚ DIỄN GIẢ (SPEAKER NOTES)

*Dưới đây là phần nội dung. Bạn hãy copy phần "Cell Markdown" dán vào ô Markdown của notebook và chọn Slide Type là `Slide`. Phần "Ghi chú" dán vào 1 ô bên dưới và chọn Slide Type là `Notes`.*

---

### [Cell Markdown: Tùy chỉnh Title Slide]
# GIẢI PHÁP KHÔI PHỤC VÀ TĂNG CƯỜNG CHẤT LƯỢNG HÌNH ẢNH TÀI LIỆU QUÉT TỪ CAMERA DI ĐỘNG
**Document Restoration & Enhancement ML Pipeline**

- **Môn học:** Xử lý Ảnh và Video
- **Nhóm thực hiện:** Nhóm 6
- **Giáo viên hướng dẫn:** [Tên Giáo Viên]

### [Diễn giả - Notes]
> Chào mừng thầy/cô và các bạn. Hôm nay, Nhóm 6 xin trình bày đề tài mang tính ứng dụng thực tiễn rất cao: Giải pháp khôi phục và xử lý ảnh chụp tài liệu từ Camera điện thoại thành định dạng như quét từ máy tính chuyên dụng. Bài toán này lấy cảm hứng từ các siêu ứng dụng như CamScanner hay Microsoft Lens.

---

### [Cell Markdown: Agenda]
## NỘI DUNG BÁO CÁO

1. **Đặt vấn đề:** Thách thức vật lý khi chụp ảnh tài liệu bề mặt.
2. **Tổng quan nghiên cứu:** Máy quét truyền thống vs. AI hiện đại.
3. **Phạm vi & Đề xuất (Pipeline Nhóm 6):** Sự kết hợp giữa AI và Computer Vision.
4. **Kết quả thực nghiệm:** Phân tách và So sánh hiệu năng.
5. **Kết luận & Hướng phát triển.**

### [Diễn giả - Notes]
> Bài báo cáo của nhóm sẽ đi qua 5 phần chính: Mở đầu bằng sự kiện các thách thức vật lý để làm rõ lý do tại sao ứng dụng Scanner thực thụ lại khó làm. Kế tiếp, chúng em sẽ đề xuất một Pipeline lai (Hybrid) kẹp giữa Machine Learning và Xử lý ảnh OpenCV mà nhóm đã tự xây dựng. Về cuối sẽ là minh họa kết quả thị giác thực tế.

---

### [Cell Markdown: Slide 3 - Vấn đề]
## ĐẶT VẤN ĐỀ: Thách Thức Khi Chụp Bằng Điện Thoại

Khác với máy quét cố định, camera di động phải đối mặt với **4 nhóm rào cản vật lý** tàn khốc:

1. **Giao thoa Ánh sáng:** Đổ bóng tay/điện thoại, ánh sáng lóa huỳnh quang, đốm chói Flash (Glare), hoặc chụp thiếu sáng gây nhiễu ISO.
2. **Biến dạng Hình học:** Tư thế cầm chéo làm méo phối cảnh (Perspective). Góc giấy quăn hoặc biến dạng cong vút lồi lõm vùng gáy sách.
3. **Cơ cấu Camera:** Rung mờ chuyển động (Motion Blur), mất tâm lấy nét, hoặc hạt pixel quá gắt gây nhiễu răng cưa (Moire) trên lưới nền.
4. **Bề mặt tài liệu:** Giấy cũ đóng mốc vàng, mực phai đứt nét, hoặc nét mực mặt sau lặn đè lên (Bleed-through).

### [Diễn giả - Notes]
> Khác với một cái máy quét phẳng lì nằm im trong phòng tối. Khi người dùng cầm smartphone, sẽ có cái bóng đen của chính cái điện thoại đổ lên giấy. Góc chụp ngang làm méo dẹt cái hóa đơn. Sự rung lắc của cổ tay phá vỡ nét chữ. Những 20 tiểu thách thức vật lý bao hàm thành 4 nhóm này khiến việc dùng thuật toán cổ điển để cắt chữ gặp giới hạn không thể vượt qua.

---

### [Cell Markdown: Slide 4 & 5 - Giới hạn Truyền thống]
## GIỚI HẠN CỦA PHƯƠNG PHÁP XỬ LÝ ẢNH TRUYỀN THỐNG

### Sự yếu kém của Phát hiện Biên (Canny + approxPolyDP)
- Chỉ hoạt động khi: Nền trơn màu, tờ giấy phải là đa giác **thẳng tắp 4 cạnh**.
- **Thất bại:** Khi tờ giấy bị nhàu cụp (cong), tay che khuất 1 mép, nền có nhiều chi tiết như trên thảm cỏ $\rightarrow$ Tỷ lệ thành công chỉ **25%** ở môi trường phức tạp.

### Suy giảm màu sắc từ các bộ xén Tương Phản (Thresholding)
- **Otsu (Toàn cục):** Nếu tài liệu bị dính bóng râm, vùng râm bị biến thành các tảng mù đen kịch vì dưới ngưỡng sáng.
- **Adaptive Threshold (Cục bộ):** Giải được bóng râm nhưng xén quá mạnh làm các nét chữ thanh mảnh đứt gãy gồ ghề (nham nhở viền chữ).

### [Diễn giả - Notes]
> Tại sao nhóm không dừng lại ở xử lý OpenCV truyền thống? Lý do là Canny Edge tìm 4 góc giấy rất "cứng nhắc". Ném tờ hóa đơn lên giường nệm nhiều hoa văn, hay tờ hóa đơn bị quăn mép gió bay, thuật toán xấp xỉ đa giác sẽ đứt bóng ngay. Tương tự, dùng Adaptive Thresholding bẻ trắng đen thì chữ hay bị rỗ như răng cưa, gai viền rất đau mắt khi đọc PDF lâu.

---

### [Cell Markdown: Slide 6 & 7 - Pipeline Đề xuất]
## ĐỀ XUẤT CỦA NHÓM: FRONT-LINE MACHINE LEARNING PIPELINE
Hệ thống kết hợp sức mạnh xuyên thấu của AI ở lớp cắt nền và độ tinh tế của Computer Vision ở lớp hoàn thiện bề mặt:

* **Bước 1 [ML Detection]:** 
   - Loại bỏ Canny. Dùng AI (`DocAligner` / `U²-Net`) trích xuất biên chính xác 100%.
   - Đặc biệt: Khởi động `U²-Net` khoét xuyên khung nền nếu giấy bị lượn sóng không định hình.
* **Bước 2 [Geometric Dewarping]:** 
   - Biến đổi Phối cảnh (kéo hình thang) + Trí tuệ Nắn dòng chữ (`page-dewarp` / `UVDoc`) bẻ phẳng độ cong của sách.
   - ⚡ **Toán Học Giám Sát (Geometric IoU):** Khóa an toàn chống Méo (Anti-Pinch). Tự động dùng Toán học phân tích viền lượn sóng so với viền thẳng lý tưởng để khóa mạng UVDoc nếu phát hiện giấy đã xếp phẳng (>94%).
* **Bước 3 [CV Enhancement]:** Tăng cường chốt chặn (Khử lóa $\rightarrow$ Kéo viền rung $\rightarrow$ Phơi sáng xám bù bóng). 🌟 Đặc biệt: **Khử Bóng Râm Độc Lập 3 Kênh RGB**, phục hồi sắc trắng nhưng lưu giữ trọn vẹn màu mực Dấu Đỏ, Chữ Ký Xanh không bị cháy xám (Color Halos).

### [Diễn giả - Notes]
> Từ những yếu điểm trên, nhóm đã nảy ra ý tưởng xây dựng 1 Pipeline lai (Hybrid). Thay vì vất vả dùng Toán học Canny dò từng mảnh rìa dễ đứt gãy để lấy góc, nhóm đánh thẳng AI cực mạnh như DocAligner hoặc U²-Net vào Bước 1 để não bộ AI hiểu "đâu là tờ giấy". Nhưng AI cũng có thể ảo giác làm méo giấy phẳng, nên ở Bước 2 nhóm lập chốt chặn Toán Học tính diện tích IoU để giám sát gắt gao. Cuối cùng ở Bước 3, OpenCV can thiệp nâng cấp ánh sáng bằng cơ chế Tách Kênh RGB độc lập, đảm bảo nền trắng tinh nhưng con dấu đỏ vẫn rực rỡ y như thật.

---

### [Cell Markdown: Slide 8 - Đỉnh Cao Tăng Cường]
## ĐIỂM SÁNG CÔNG NGHỆ: ADAPTIVE CLAHE \u0026 BÁCH PHÂN VỊ HISTOGRAM

**Tại sao bức ảnh xuất ra từ Pipeline của nhóm lại đanh nét như PDF In từ máy tính?**
- Hệ thống **TỪ CHỐI** sử dụng Cắt ngưỡng (Threshold) cố định.
- Áp dụng kỹ thuật: **Tự thích ứng cường độ mảng (Adaptive CLAHE).**
  - **Tương Phản Lưới:** Mạng `CLAHE` quét trên lưới 8x8 lôi bật nét viết nhạt phai nằm lọt thỏm trong bóng tối đen.
  - **Trích Xuất % Toán Học:** Kéo Bảng Histogram thành mảng 1D, tự động nội suy lấy $2\%$ tín hiệu đen nhánh nhất làm điểm neo chữ; và $95\%$ làm điểm rực trắng đẩy bức ảnh bốc hơi giấy dính rác ố.
  - Hỗ trợ khử nhiễu đa kênh **LAB** cho file màu, hoàn toàn tẩy bay đốm xám liti trên nền giấy trắng.

### [Diễn giả - Notes]
> Và điểm chốt hạ tạo ra sự khác biệt về mặt thị giác tốt nhất là Bước Phơi Sáng Tự Thích Ứng chặng cuối. Khác với Otsu hay ngưỡng cố định thường làm cháy mất chữ mờ nếu sáng tối chênh lệch. Nhóm thiết kế một bộ quét mạng phân vùng lưới CLAHE để soi rọi góc tối, kết hợp việc tự trích xuất Histogram Bách Phân Vị thông minh. Kết quả chữ sẽ đen tuyền, nền trắng phau cực kì cuốn và êm mắt, dù là tờ hóa đơn rách nát hay hợp đồng phẳng lì.

---

### [Cell Markdown: Slide 9 - Thực nghiệm 1]
## THỰC NGHIỆM: CHIẾN THẮNG TRƯỚC RÀO CẢN HÌNH HỌC (BƯỚC 1 & 2)

*Hãy chạy Code / Show ảnh ở slide này trong Notebook*
1. **Phân vùng (Segmentation):** Xử lý hoàn hảo nền giả thảm len rườm rà.
2. **Khôi Phục Phối Cảnh:** Thuật toán chiếu ma trận lật dọc trang giấy từ ảnh méo xẹo.
3. **Nắn thẳng gáy (Text-line Dewarping):** Dòng chữ trên cuốn sách bị võng (Curved) đã được U²-Net bóc nền bảo toàn nếp cong, và đưa qua module ủi phẳng (Flatten) ngay lập tức.

### [Diễn giả - Notes]
> Thầy và các bạn xem hình thực nghiệm trên slide. Giấy cong lượn quăn mép không còn là vấn đề khi AI nó nhận diện nguyên cái cục giấy đó thay vì ép phải tìm đủ 4 góc vuông. Cụm dewarping của hệ thống đã nội suy cái bề mặt 3D đó và ép nó thành 1 file 2D phẳng như mặt bàn.

---

### [Cell Markdown: Slide 10 - Thực nghiệm 2]
## THỰC NGHIỆM: XÓA LÓA & KHỬ RUNG (BƯỚC 3a, 3b)

- **Vá lóa (In-painting):** Đốm đèn Flash bắn thủng một lỗ mù chữ giữa tờ giấy $\rightarrow$ Hàm Inpaint sẽ thu thập dữ liệu xung quanh lỗ thủng để dệt kín lại, hạ sáng vùng quét.
- **Motion Blur (Khử rung):** Tay thu hoạch hơi lắc khiến cạnh mép chữ nhòe tan. Blur Unsharp Masking lấy ảnh phân rã đẩy biên độ để mép chữ dằn sắc nhọn như dao cạo, trả lại sự rõ nét quang học.

### [Diễn giả - Notes]
> Hình ảnh này là đốm flash hay làm mù mất mấy chữ. Chúng em dùng Inpaint bù ảnh. Còn ở ảnh bên cạnh, chữ bị nhòe rông do máy out-net, đã được bộ lọc thông cao đẩy gắt rìa chữ, lấy lại các chân nối sợi mực.

---

### [Cell Markdown: Slide 11 & 12 - Thực nghiệm 3]
## THỰC NGHIỆM: CHỐNG BÓNG RÂM ĐỘC LẬP MÀU & PHƠI SÁNG (BƯỚC 3c, 3d)

- **Khử Bóng Râm Độc Lập Kênh Màu (LAB Denoise \u0026 RGB Division):** Ánh sáng bóng đổ (tay người) dội qua thuật toán chia màng Lưới chiếu sáng nhân tạo (Illumination). Đặc biệt, file màu được nhúng qua bộ khử **LAB Denoising**, triệt tiêu toàn bộ rác tạp xám, tím li ti. Con Dấu đỏ và Mực xanh nguyên bản rực rỡ y hệt file Vector Digital gốc, trong khi tờ giấy trắng phau.
- **Tự Phân Xác Bách Phân Vị (Adaptive Binarize):** Độ nét con chữ được bọc lót chống đứt mẻ răng cưa tuyệt đối. Viền ôm hình thái cong tự nhiên. Tẩy ố mốc ngả vàng ở tài liệu lịch sử hoàn hảo 100%.

### [Diễn giả - Notes]
> Bước cuối cùng, mọi người có thể xem bóng tay đen thui ở góc giấy đã bị triệt tiêu hoàn toàn. Đặc biệt, nhóm em không ép ảnh thành Trắng đen ngay, kết hợp bộ lọc nhiễu LAB. Con Dấu đỏ và bút bi Xanh hoàn toàn giữ nguyên được độ Rực Rỡ sắc sảo, hệ thống đẩy nhẹ kênh Saturation một cách cực mượt. Viền chữ cũng không hề sứt mẻ lỗ rỗ do cắt ngưỡng bởi vì bộ đo Histogram làm việc quét đỉnh sáng - đáy tối quá ưu việt. Bức ảnh cuối cùng sẽ có chất lượng cao cấp ngang ngửa máy Scan chuyên dụng trăm triệu.

---

### [Cell Markdown: Slide 13]
## KẾT LUẬN & HƯỚNG MỞ RỘNG

**1. Kết luận:**
- Pipeline mang tính thực tiễn cao, tỷ lệ thành công tăng từ **~25%** ở OpenCV cổ điển lên sấp xỉ **>95%** nhờ mô hình lai ghép Convolution AI.
- Giải quyết ráo riết nhược điểm gây đứt gãy nham nhở của các phương pháp nhị phân hóa (Binarization) cũ nát.

**2. Hướng mở rộng:**
- Đối với bài toán có chứa hình ảnh màu trong giấy (Figure): Kế hoạch sắp tới là nhúng thêm **Layout Analysis** bóc riêng vùng ảnh màu ra không cho chạy qua máy phơi sáng đen trắng, giúp lai cả chữ rõ nét và hình rực rỡ.
- Đưa trọng lượng Neural Network lên các bản tệp vi nén (Edge Devices - TFLite) để chạy Offline tiết kiệm RAM trực tiếp trên thiết bị Mobile.

### [Diễn giả - Notes]
> Tóm lại, phương pháp dùng Deeplearning làm tiền đạo đi trước dọn vấp ngã mặt vật lý, sau đó mời Xử lý Ảnh vào thu gom chất lượng tàn dư giúp hệ thống của nhóm vô cùng bền bỉ. Ở tương lai nhóm cũng nghiên cứu việc giữ nguyên được màu sắc gốc cho hình trong giấy thay vì giã nát đen trắng để nó xịn xò như một cái máy Photocopy xịn. Em xin cảm ơn.
