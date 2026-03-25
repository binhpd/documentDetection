# KỊCH BẢN THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN (SPEAKER NOTES)
*Tài liệu này cung cấp lời thoại chi tiết cho từng Slide (từ 1 đến 17), đặc biệt lồng ghép bài toán cực kỳ kinh điển: "Làm trắng nền giấy vướng bóng râm nhưng tuyệt đối không được giết chết màu sắc của con dấu đỏ hay chữ ký xanh".*

---

### SLIDE 1: TRANG BÌA
**Lời mở đầu:** 
"Kính chào hội đồng ban giám khảo và các thầy cô. Hôm nay, đại diện Nhóm 6, em xin phép trình bày đồ án môn Xử lý Ảnh và Video với đề tài: **Giải pháp khôi phục và tăng cường chất lượng hình ảnh tài liệu quét từ thiết bị di động bằng Hybrid Machine Learning.**"

---

### SLIDE 2: NỘI DUNG BÁO CÁO (AGENDA)
**Lời thoại:** 
"Buổi báo cáo hôm nay của nhóm sẽ đi qua 5 trọng tâm chính: Khởi đầu bằng việc mổ xẻ những thách thức khắc nghiệt của môi trường vật lý. Tiếp đến là giới hạn của các công cụ cũ. Trọng tâm cốt lõi là việc nhóm lai tạo AI vào đường ống OpenCV (Hybrid Pipeline). Sau đó, nhóm sẽ đo lường hiệu năng kiến trúc, và chốt lại bằng kết quả thực nghiệm trực quan."

---

### SLIDE 3: NHÓM THÁCH THỨC VẬT LÝ CỰC ĐOAN (🔥 ĐIỂM NHẤN)
**Lời thoại:** 
"Để biến chiếc điện thoại thành một cỗ máy scan công nghiệp, chúng ta phải giải bài toán nhiễu loạn môi trường. Rung tay, giấy quăn xé góc hay thảm nền rườm rà vốn dĩ đã khó. 
Nhưng sự đánh đố lớn nhất của đồ án này nằm ở **Dòng Giao Thoa Ánh Sáng và Chênh Lệch Màu Sắc**. Khi người dùng cúi xuống chụp, bóng lưng hoặc bóng cánh tay đổ sập xuống mặt giấy tạo ra các mảng chênh sáng mù mịt hắt hiu (Illumination variance).
**Thách thức tối thượng đặt ra là:** Thuật toán bắt buộc phải bóc gỡ được lớp bóng râm đen đặc đó, đẩy nền giấy về chuẩn Trắng Sáng tinh khiết, **NHƯNG... không được tẩy bay màu sắc gốc của văn bản.** Nếu một văn bản pháp lý có Con Dấu Mộc Đỏ hoặc Chữ Ký Bút Bi Xanh mà đưa qua app Scan bị tẩy phai ra màu xám xịt hay ám đen cộc lốc, thì giá trị của bản Scan đó hoàn toàn VÔ DỤNG."

---

### SLIDE 4: TỔNG QUAN CÁC HƯỚNG NGHIÊN CỨU
**Lời thoại:** 
"Giới nghiên cứu học thuật hiện nay giải quyết vòng lặp này qua 3 chốt chặn: Đầu tiên là Localization (Tìm góc giấy bằng Toán học hoặc Mạng Phân đoạn U-Net). Thứ hai là Dewarping (Nhào nặn uốn phẳng độ cong bằng Spline 3 chiều). Và chốt hạ cuối cùng là mảng Enhancement (Binarize phân ngưỡng cường độ sáng để lột xác điểm ảnh)."

---

### SLIDE 5: GIỚI HẠN CỦA "SCANNER TRUYỀN THỐNG"
**Lời thoại:** 
"Và đây là rào cản tử huyệt của xử lý ảnh truyền thống. Nếu xài thuật toán Otsu (bổ ngưỡng toàn ảnh), nó sẽ ngay lập tức nhuộm đen xì toàn bộ những vùng bị bóng râm che lấp. 
Khôn ngoan hơn một chút, người ta chuyển sang xài thuật toán Local Adaptive (Phân ngưỡng cục bộ). Nó quét qua được lớp bóng râm, tuy nhiên lưỡi dao cắt của nó quá độc đoản. Nó bào thủng những nét chữ nhạt mảnh, làm đứt gãy vành con chữ và đặc biệt là **NÓ ÉP TỬ MÀU SẮC**. Nó tước đoạt toàn bộ màu đỏ màu xanh thành các hạt pixel đen trắng nham nhở gai góc."

---

### SLIDE 6: MỤC TIÊU PHẠM VI DỰ ÁN
**Lời thoại:** 
"Hiểu được nỗi đau đó, sứ mệnh của Pipeline Nhóm 6 cực kỳ rõ ràng: Đầu vào là một tấm ảnh thô cong vênh, rập bóng người thui lủi. Đầu ra kỳ vọng là một Text Layout quét vuông vức, nền trắng bóc trong vắt, nét mảnh đen ôm gọn gàng, và đặc biệt Con dấu đỏ vẫn phải sực mùi mực đỏ!"

---

### SLIDE 7: PHƯƠNG PHÁP TIẾP CẬN – SƠ ĐỒ PIPELINE
**Lời thoại:** 
"Để làm được điều đó, nhóm xây dựng bộ máy **Hybrid Pipeline**. 
- Bước 1: Máy đóng vai trò là Sát Thủ Tuyến Đầu. Gọi AI (U²-Net hoặc DocAligner) tước lột thảm phông nền.
- Bước 2: Dùng ma trận Perspective nắn chéo, và cầu viện AI UVDoc nắn thẳng dòng uốn lượn lõm gáy sách.
- Bước 3: Đẩy về cho Toán học Không Gian OpenCV cầm cương dồn dập 4 lõi đả kích sắc độ."

---

### SLIDE 8: CỐT LÕI ĐỘT PHÁ TĂNG CƯỜNG ÁNH SÁNG & BẢO TOÀN MÀU
**Lời thoại:** 
"Xin phép hội đồng chú ý kỹ vào Thuật toán khâu Tăng cường màu sắc (Bước 3). Thay vì gộp bức ảnh về thang độ Xám rồi cày nát điểm ảnh, nhóm triển khai **Giải pháp Triệt Bóng Kênh Độc Lập RGB**.
Chúng em tách mạch máu bức ảnh thành 3 kênh riêng biệt Đỏ, Lục, Lam. Thuật toán sẽ đúc một màng Gaussian làm giả mạo lớp bóng râm cho từng kênh. Sau đó lấy Ảnh Gốc Đem CHIA cho Ảnh Giả Mạo. 
Nhờ cơ chế Tách Dòng (Channel-wise Division), vùng nền mờ mịt bị trung hòa đánh rỗng bốc hơi thành màu Trắng phau, **nhưng** đặc tính tần số thấp của Mực Đỏ và Ký Xanh không hề chạm ngưỡng bão hòa nên vẫn ngời ngợi chân thực 100%."

---

### SLIDE 9: SO SÁNH GIẢI PHÁP NHÓM VS. CV TRUYỀN THỐNG
**Lời thoại:** 
"Do đó, khác với bộ định tuyến Canny (OpenCV gốc) dễ dàng bị chấn thương đứt đoạn bởi cái bàn có vân hoa sen rườm rà, Pipeline của nhóm gắp tờ giấy ra khỏi không gian nhờ "phân tích ngữ nghĩa" (đứa trẻ hiểu tờ giấy). Và khác với thuật toán cũ vắt kiệt sinh học màu sắc (Monochrome), giải pháp nhóm giữ trọn sinh khí của màu sắc sặc sỡ trên tài liệu."

---

### SLIDE 10: SO SÁNH CHUYÊN SÂU KHỐI APP THƯƠNG MẠI
**Lời thoại:** 
"Nhìn sang các đế chế ứng dụng hiện hành như CamScanner hay Adobe Scan, họ có lợi thế là tốc độ cực kì bay bổng vì thuần túy xài CV (Toán ròng). Tuy nhiên, họ lại thất thủ hoàn toàn trước một quyển sách dày cộm lượn sóng võng gáy (do không có tư duy mạng Neuron cong 3D) và họ đang lạm dụng dập Binarize gắt mạnh bạo cắn mất nét chữ phai nhạt."

---

### SLIDE 11: ĐÁNH GIÁ MỨC TỐI ƯU CỦA THIẾT KẾ CẤU TRÚC
**Lời thoại:** 
"Thiết kế của nhóm đảm bảo sự giao thoa quyền lực tối ưu: Việc nhận thức sự méo mó, quăn góc, lượn sóng (Cái khó vĩ mô) dâng lên cho Trí Thuệ Nhân Tạo. Còn việc đan vá điểm ảnh xé mổ màu sắc (Cái khó vi mô) được OpenCV làm chủ vì chỉ có Toán Học gốc mới không đi "bù nhìn ảo giác". 
Đặc biệt, hệ thống có **Rào Chắn Diện tích IoU**. Đưa tờ rơi phẳng vào, nó gạt AI qua một bên, áp chóp 4 góc siêu tốc. Đưa sách cong vào, nó mới mở vách ngăn AI kéo phẳng. Tiết kiệm hệ số tính toán rực rỡ!"

---

### SLIDE 12: PHÂN TÍCH CHI PHÍ ĐIỆN TOÁN VÀ KHẢ THI ON-DEVICE
**Lời thoại:** 
"Chi phí đắt đỏ của Deep Learning có phải là mộng tưởng trên Smartphone? Không thưa thầy cô! Khâu tách nền hiện nay xài NPU của Nano chỉ lốn $< 50ms$. Khâu nắn sóng ResNet UVDoc tương lai lượng tử hóa qua TFLite dung lượng nằm nệm dưới 40MB. Còn mảng mổ xẻ màu sắc của Bước 3 toàn tệp số NumPy vector hóa qua C++ Backend ngốn không tới 30ms. Khả năng bó cục hệ thống thành một App Offline 80 Megabytes không chạy Cloud là điều ở ngay trước mắt."

---

### SLIDE 13: KẾT QUẢ THỰC NGHIỆM STEP 1 & 2
**Lời thoại:** 
"Đến với phần thực nghiệm: Hình 1 là giấy dán trên một thảm len bề bộn rác chụp trong môi trường phòng tối. Trải qua lớp lọc đầu tiên, nó khước từ hoàn toàn vân len cuộn, gạt tay người chặn giấy, bắn vút vào hệ thống lưới để ép mép quăn mượt mà như 1 dao cạy."

---

### SLIDE 14: KẾT QUẢ THỰC NGHIỆM ĐÁNH LÓA & RUNG NHÒE
**Lời thoại:** 
"Đây là zoom cận cảnh Macro điểm ảnh. Tia đèn flash trắng khoét một lỗ kim chói giữa mép chữ. Phép Inpainting lập tức túm thịt xung quanh vá kín miệng vết thương. Kế bên, vết quẹt nét rung tay bay màu đuôi chữ được đập dập, cắt gọn cạo vành sắc tứa mạnh bạo (Unsharp Masking)."

---

### SLIDE 15: KẾT QUẢ TRIỆT TIÊU BÓNG LOANG (BẢO TOÀN MÀU CỐT LÕI)
**Lời thoại:** 
"Đây là thành tựu cốt lõi nhất hồi nãy em có đề cập: Ở bức hình bên trái, mảng bóng đổ tay người hắt tối rù rì chím lấp một nửa sấp văn bản. 
Và bum! Ở hình bên phải, với Division Shadow Normalization, ánh sáng được thổi tung tản băng 100% tờ giấy. Chữ đen nháy. Và quý thầy cô hãy zoom thẳng vào **con dấu đỏ thẫm** và **mực ký xanh**. Tông màu nguyên thủy cực kỳ rực rỡ. Không có bất cứ 1 vết răng cưa đen xám hay ám tím ảo ảnh (color halo) nào đọng lại quanh chữ ký. Chuẩn pháp lý cực kỳ nguyên vẹn!"

---

### SLIDE 16: ĐỈNH CHÓP SỨC MẠNH KÍCH MỀM BINARIZE
**Lời thoại:** 
"Khi soi cặn kẽ nét chữ binarize (ảnh xám đen): Ở mép ngoài, đường rãnh Binarize Otsu xé rỗ thủng rạn nứt cọc cạch. Còn bên phương án Soft-Thresholding của nhóm, vành nét chữ được bọc một con đệm sương mờ Anti-Aliasing (xám tuyến tính) bao thắt bụng mực cốt đen tuyền. Chữ rất mượt mà trong trạng thái siêu rõ."

---

### SLIDE 17: KẾT LUẬN TỔNG THỂ & HƯỚNG MỞ QUY MÔ
**Lời thoại:** 
"Tóm lại: Đồ án là minh chứng cho sự ưu việt của việc Không Giam Hãm bản thân vào 1 phe rập khuôn rỗng tuếch. Tái cơ cấu AI thay cho mặt trận dò góc, và Tôn tạo giữ nguyên OpenCV mài dũa độ nét (Bảo tồn sinh học màu sắc), đồ án đã tạo ra một buồng máy Scan thực chiến kinh hoàng. Hướng tiếp tục của nhóm: Lượng tử hóa file Nơ-ron và bó trọn vào lõi Di Động App Native."
