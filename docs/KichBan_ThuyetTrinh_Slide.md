# KỊCH BẢN THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN (SPEAKER NOTES)
*Tài liệu này cung cấp lời thoại chi tiết cho từng slide (từ 1 đến 17), tập trung vào bài toán cốt lõi: "Làm trắng nền giấy bị khuất bóng nhưng vẫn giữ nguyên màu sắc chân thực của con dấu đỏ và chữ ký xanh".*

---

### SLIDE 1: TRANG BÌA
**Lời mở đầu:** 
"Kính chào hội đồng ban giám khảo và các quý thầy cô. Hôm nay, đại diện Nhóm 6, em xin phép trình bày đồ án môn Xử lý Ảnh và Video với đề tài: **Giải pháp khôi phục và tăng cường chất lượng hình ảnh tài liệu quét từ thiết bị di động bằng Hybrid Machine Learning.**"

---

### SLIDE 2: NỘI DUNG BÁO CÁO (AGENDA)
**Lời thoại:** 
"Buổi báo cáo hôm nay của nhóm sẽ đi qua 5 nội dung chính: Đầu tiên là phân tích những khó khăn của môi trường chụp ảnh thực tế. Tiếp đến là những hạn chế của các phương pháp xử lý truyền thống. Trọng tâm của nhóm là sự kết hợp giữa mô hình machine learning và các thuật toán chuyên sâu về xử lý ảnh số, được xây dựng và phát triển dựa trên nền tảng thư viện mã nguồn mở OpenCV (mô hình Hybrid Pipeline). Sau đó, nhóm sẽ đánh giá hiệu năng của hệ thống, và cuối cùng là các kết quả thực nghiệm đạt được."

---

### SLIDE 3: CÁC NHÓM THÁCH THỨC TỪ MÔI TRƯỜNG THỰC TẾ
**Lời thoại:** 
"Để biến chiếc camera điện thoại thành một máy scan đúng nghĩa, thực tế chúng ta phải giải quyết **4 nhóm thách thức vật lý cốt lõi**:
Thứ nhất là **Biến dạng hình học**: Từ việc ảnh bị lệch góc phối cảnh, mép giấy quăn, bề mặt nhăn nheo lồi lõm cho đến độ uốn cong phức tạp của gáy sách.
Thứ hai là **Sai hỏng tiêu cự và Rung máy**: Gồm tình trạng ảnh bị rung mờ (motion blur) do thao tác tay hay bị nhòe do thiết bị mất điểm lấy nét.
Thứ ba là **Sự xuống cấp bề mặt tài liệu**: Chẳng hạn như nền giấy ố vàng, nét mực phai mờ đứt đoạn, hay vết hằn mực từ mặt sau xuyên qua giấy mỏng.
Và cuối cùng - cũng là **khó khăn lớn nhất** thuộc nhóm **Giao thoa ánh sáng**: **Hiện tượng phân bố ánh sáng không đồng đều và sai lệch màu sắc**. Khi người dùng thao tác, bóng đổ của cơ thể, vết lóa của đèn huỳnh quang hay đốm chói lóa từ đèn flash (glare) thường che khuất mặt giấy.
**Yêu cầu cực kỳ khắt khe đặt ra là:** Thuật toán phải xử lý triệt để lớp bóng râm và các vùng chói này, đưa nền giấy về chuẩn trắng sáng, **nhưng tuyệt đối không được làm ảnh hưởng đến màu sắc gốc của văn bản.** Việc bảo toàn trọn vẹn màu sắc của con dấu đỏ hay chữ ký xanh là điều kiện tiên quyết để văn bản quét giữ được tính nguyên bản và giá trị pháp lý sử dụng."

---

### SLIDE 4: TỔNG QUAN VÀ LÝ GIẢI HƯỚNG TIẾP CẬN
**Lời thoại:** 
"Để giải quyết các bài toán trên, hiện nay có hai hướng tiếp cận phổ biến: Một là dùng các công vụ xử lý ảnh truyền thống (như OpenCV), ưu điểm là tính toán cực nhanh nhưng dễ thất bại khi nền phức tạp; Hai là sử dụng mô hình Học sâu (Deep Learning) toàn trình, rất thông minh trong nhận diện không gian nhưng lại hay làm biến dạng màu sắc và tính toán nặng nề. 
Nhằm khắc phục các khuyết điểm trên, nhóm đã **lựa chọn hướng tiếp cận Lai (Hybrid Pipeline)**. Quyết định này kết hợp khả năng phân tích bối cảnh xuất sắc của AI và sự tinh tế trong việc tinh chỉnh màu sắc của các thuật toán xử lý ảnh số. Cụ thể, giải pháp đi qua 3 bước chuẩn mực: **Localization** (AI bóc tách nền), **Dewarping** (Cân chỉnh phối cảnh và nắn phẳng tài liệu cong) và kết thúc ở **Enhancement** (Tối ưu ánh sáng và phục hồi màu sắc gốc)."

---

### SLIDE 5: Giới hạn của sử dụng TRUYỀN THỐNG OPENCV
**Lời thoại:** 
"Ở giai đoạn đầu, nhóm đã thử nghiệm xử lý bài toán hoàn toàn bằng các thuật toán có sẵn của OpenCV. Thực tế ghi nhận giải pháp này chỉ đáp ứng tốt ở các điều kiện lý tưởng cực kỳ đơn giản: ví dụ như nhận diện một tờ giấy trắng tinh đặt trên một nền bàn trống trơn tương phản mạnh. 
Tuy nhiên, khi đưa vào môi trường thực tế nhiễu loạn với nhiều góc độ và ánh sáng khác nhau, OpenCV thuần túy bộc lộ 3 nhược điểm chí mạng:
Thứ nhất, thuật toán ngưỡng toàn cục (Otsu) lập tức thất bại. Các vùng chênh sáng bị bóng râm sẽ biến thành mảng nhiễu đen thui che lấp toàn bộ chữ.
Thứ hai, nếu đổi sang ngưỡng cục bộ (Local Adaptive) để khắc phục bóng râm, thì viền nét cắt lại quá cứng nhắc, cắn xén đứt gãy các nét chữ ký mảnh.
Và yếu huyệt lớn nhất thứ ba: Các phép toán trên đều ép bức hình phải chuyển về hệ nhị phân đen/trắng. Điều này đồng nghĩa với việc **đánh mất hoàn toàn màu sắc chân thực** của tài liệu — khiến con dấu đỏ hay chữ ký xanh bị tẩy màu xám xịt vô hồn."

---

### SLIDE 6: MỤC TIÊU VÀ PHẠM VI DỰ ÁN
**Lời thoại:** 
"Từ những thách thức trên, mục tiêu của dự án Nhóm 6 được định hình rất rõ ràng: Nhóm khao khát xây dựng một hệ thống giải pháp có khả năng nhận diện và căn chỉnh tự động tài liệu thông qua hình ảnh thu thập trực tiếp từ thiết bị di động cá nhân (Smartphone). 
Yêu cầu cốt lõi là hệ thống phải xử lý tốt trong đa dạng các điều kiện ánh sáng và các tình trạng vật lý phức tạp của giấy tờ (nhăn, cong vênh, bóng đổ rải rác). 
Đặc biệt, quy trình này không chỉ **tăng cường sắc nét độ tương phản** của chữ viết, mà còn bắt buộc **vẫn giữ gìn nguyên vẹn đặc trưng nét chữ, màu mực gốc và cái hồn chất liệu (texture) nguyên bản của mặt giấy.**
Kết quả chuyên nghiệp đầu ra luôn hướng tới mục đích cao nhất: Đáp ứng xuất sắc toàn bộ quy chuẩn của công tác **số hóa chuyển đổi điện tử và lưu trữ tài liệu giấy tờ hàng ngày** trong đời sống và văn phòng."

---

### SLIDE 7: PHƯƠNG PHÁP TIẾP CẬN – SƠ ĐỒ PIPELINE
**Lời thoại:** 
"Để đạt được mục tiêu này, nhóm xây dựng mô hình vòng lặp kết hợp **Hybrid Pipeline**. 
- Bước 1: Sử dụng AI (như U²-Net hoặc DocAligner) để bóc tách tài liệu khỏi nền xung quanh.
- Bước 2: Dùng ma trận phối cảnh (Perspective) để căn chỉnh phẳng, và kết hợp mô hình AI UVDoc để làm phẳng các tài liệu bị cong lượn như trang cuốn sách.
- Bước 3: Áp dụng các thuật toán xử lý không gian màu của OpenCV để tối ưu hóa ánh sáng và tái tạo màu sắc."

---

### SLIDE 8: ĐIỂM NHẤN TRONG TĂNG CƯỜNG ÁNH SÁNG & BẢO TOÀN MÀU
**Lời thoại:** 
"Xin phép hội đồng chú ý vào thuật toán tăng cường màu sắc ở Bước 3. Thay vì chuyển bức ảnh về ảnh xám thông thường, nhóm áp dụng **giải pháp xử lý bóng đổ trên từng kênh màu RGB độc lập**.
Nhóm tách hình ảnh thành 3 kênh riêng biệt: Đỏ, Xanh lá, Xanh dương. Sử dụng bộ lọc làm mờ diện rộng (Gaussian) để ước lượng vùng bóng râm cho từng kênh, sau đó chuẩn hóa bằng cách chia ảnh gốc cho khung bóng râm vừa nội suy.
Nhờ cơ chế này (Channel-wise Division), vùng nền tối được làm sáng và trở nên đồng đều, **trong khi** các chi tiết màu như các mộc đỏ hay chữ ký xanh vẫn giữ được độ sắc nét và màu sắc một cách độc lập chân thực."

---

### SLIDE 9: SO SÁNH GIẢI PHÁP NHÓM VS. XỬ LÝ ẢNH TRUYỀN THỐNG
**Lời thoại:** 
"So với phương pháp dò biên cạnh Canny của OpenCV truyền thống, vốn dễ bị nhiễu do phông nền đồ vật lộn xộn, thuật toán của nhóm có khả năng nhận diện vùng tài liệu hữu hiệu nhờ việc phân tích ngữ cảnh của mạng học sâu. Hơn nữa, thay vì chuyển đổi sang ảnh đơn sắc làm mất bản sắc gốc của tài liệu, giải pháp của nhóm ưu tiên duy trì thông tin màu sắc trọn vẹn của văn bản."

---

### SLIDE 10: SO SÁNH VỚI CÁC ỨNG DỤNG THƯƠNG MẠI
**Lời thoại:** 
"Khi so sánh với một số ứng dụng hiện hành như CamScanner hay Adobe Scan, các ứng dụng này có ưu thế về tốc độ xử lý nhanh. Tuy nhiên, chúng có thể gặp khó khăn với các tài liệu độ cong lớn (như gáy sách cuốn) hoặc thường áp dụng độ tương phản quá mạnh (Binarize gắt), dẫn đến việc đôi khi các nét mực bị mất đi."

---

### SLIDE 11: ĐÁNH GIÁ MỨC TỐI ƯU CỦA THIẾT KẾ
**Lời thoại:** 
"Thiết kế của nhóm đạt được sự phân bổ công việc hiệu quả: Các mô hình AI đảm nhận việc nhận diện hình dáng và xử lý độ uốn cong phức tạp, trong khi mảng kiến thức xử lý ảnh số truyền thống tinh chỉnh tối ưu ánh sáng và phục hồi màu sắc mảnh. 
Ngoài ra hệ thống cũng linh động đánh giá tình trạng tài liệu. Với tài liệu thẳng phẳng, hệ thống xử lý cắt góc nhanh chóng; chỉ với tài liệu độ võng cao, mô hình AI nắn cong mới được kích hoạt, điều này giúp tối ưu hóa hệ số chi phí tính toán."

---

### SLIDE 12: ĐÁNH GIÁ CHI PHÍ ĐIỆN TOÁN VÀ KHẢ THI TRÊN THIẾT BỊ DI ĐỘNG
**Lời thoại:** 
"Về tính khả thi để hoạt động trơn tru trên thiết bị cá nhân, các khâu xử lý của hệ thống hiện tại rất ấn tượng. Khâu tách nền trên thiết bị di động chỉ mất khoảng dưới 50ms. Mô hình làm phẳng UVDoc có thể được lượng tử hóa (quantization) xuống dưới quy mô 40MB. Phương pháp hiệu chỉnh màu sắc đa phần tính toán ma trận với thời gian chưa tới 30ms. Đưa toàn bộ module này thiết lập thành ứng dụng ngoại tuyến (offline) dung lượng nhỏ là mục tiêu kỹ thuật hoàn toàn khả thi."

---

### SLIDE 13: KẾT QUẢ THỰC NGHIỆM - BƯỚC 1 & 2
**Lời thoại:** 
"Xin mời thầy cô quan sát kết quả thực nghiệm: Hình bên trái là tài liệu đặt trên nền thảm xáo trộn và có nhiễu bởi tay người dùng. Trải qua các thuật toán căn chỉnh, hệ thống đã loại bỏ được hậu cảnh dư thừa, bỏ qua tay bị lẹm trên góc, và căn chỉnh tự động tài liệu về mặt phẳng vuông trịa."

---

### SLIDE 14: KẾT QUẢ THỰC NGHIỆM - XỬ LÝ CHÓI MÀU & RUNG NHÒE
**Lời thoại:** 
"Tiếp theo là đánh giá cận cảnh phần nét chữ. Đối với các vùng bị loá do đèn flash thiết bị (chói khoét trắng), hệ thống áp dụng kỹ thuật Inpainting để phục hồi dữ liệu bị hư hại. Tại vùng lỗi nét chữ rung nhòe, hệ thống tăng độ sắc viền nét (Unsharp Masking) giúp chữ đậm và rõ ràng trở lại."

---

### SLIDE 15: KẾT QUẢ TRIỆT TIÊU BÓNG LOANG VÀ BẢO TOÀN MÀU
**Lời thoại:** 
"Đây là kết quả trọng điểm của dự án: Ở bức hình bên trái, mảng tối từ tay người đã hắt lệch ánh sáng tạo bóng lên văn bản. 
Nhìn sang bức ảnh bên phải, biểu đồ bóng râm được chuẩn hoá thành công mang lại diện mạo trắng đồng đều. 
Đồng thời, khi quan sát kỹ **họa tiết con dấu đỏ** và **nét mực ký xanh**, màu sắc hiển thị rất tự nhiên, rực rỡ và giữ nguyên hình thái gốc; không bị lẫn mảng tối, đốm xám hay hiện tượng loang màu."

---

### SLIDE 16: CHI TIẾT ĐỘ MƯỢT NÉT CHỮ
**Lời thoại:** 
"Khi quan sát kỹ hơn chất lượng đường nét ký tự: Các phương pháp nhị phân đơn thuần thường làm nét chữ bị răng cưa, gai góc. Còn với tiếp cận ngưỡng mềm (Soft-Thresholding) của nhóm, viền chữ được bao bọc sự chuyển tiếp mềm mại (Anti-Aliasing). Nhờ đó tạo ra đường nét mượt mà và cảm nhận nguyên bản như bản in."

---

### SLIDE 17: KẾT LUẬN & HƯỚNG PHÁT TRIỂN
**Lời thoại:** 
"Trình bày trên cho thấy hệ thống đồ án đã tối ưu khả năng kết hợp đa phương pháp: Sự ưu việt của mạng học sâu trong việc nhận dạng ranh giới cùng kỹ năng tinh giảm ảnh hưởng mảng sáng-tối từ OpenCV.
Mục tiêu tương lai của nhóm sẽ là nén hiệu suất các mô hình thuật toán học sâu để chạy đa luồng linh hoạt và tích hợp tất cả vào phần mềm máy quét đi động trực tiếp (App native)."
