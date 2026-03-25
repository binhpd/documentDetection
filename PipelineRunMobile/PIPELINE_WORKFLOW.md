# Kiến Trúc PipelineRunMobile (On-Device)

Thư mục này chứa toàn bộ mã nguồn thực thi độc lập cho một luồng quét tài liệu (Document Scanning) mô phỏng môi trường Mobile (chạy On-device 100% không cần Server).

## Tổng Hợp Luồng Công Việc (Workflow)

Pipeline này thực thi tự động qua 3 giai đoạn được thiết kế đặc biệt nhằm ép dung lượng App xuống dưới **50MB**, đồng thời duy trì độ nét 3D của AI:

### Giai Đoạn 1: Document Segmentation (Cắt Nền Tốc Độ Cao)
- **Classic Routine**: Sử dụng thuật toán truyền thống `OpenCV C++ (Canny + approxPolyDP)`. Chi phí 0MB, tốc độ <10ms trên CPU di động.
- **AI Fallback Routine**: Nếu ảnh quá phức tạp, tự động chuyển sang sử dụng mạng nơ-ron nhẹ **U²-NetP (~4.7MB)** (hoặc YOLOv8 Nano) để nội suy và khoét chính xác bóng tài liệu ra khỏi nền rác.

### Giai Đoạn 2: Neural Dewarping (Nắn Cong Giấy Lưới Không Gian)
- Bỏ qua các thuật toán biến đổi phối cảnh thẳng (Perspective Transform) thông thường.
- **Tiến trình**: Feed ảnh đã được bóc nền hoặc cắt viền vào mạng **UVDoc Neural Grid phiên bản Quantized FP16**.
- **Cách thức hoạt động**: AI tính toán hàng ngàn điểm Point Positions 2D/3D (Tạo lưới Mesh), sau đó nội suy Bilinear Unwarping để là phẳng từng điểm ảnh lồi lõm của tờ giấy.
- Dung lượng thành phần: **~15.4 MB** (thay vì 105MB bản gốc).

### Giai Đoạn 3: Hình Nền & Tiền Xử Lý Binarization
Chỉ sử dụng OpenCV C++ thuần túy nhằm tiết kiệm tuyệt đối RAM điện thoại (Vốn không thể tải thêm các mạng AI Enhance nặng nề):
- Vận dụng chia khối **Background Division** (Ảnh gốc / Ảnh Blur) để loại bỏ đổ bóng tay cầm, bù trừ bóng râm.
- **Morphology & Otsu Threshold**: Cân bằng phơi sáng mềm để cho ra con chữ đen rõ nét trên nền chuẩn trắng hoàn toàn mà không làm rách hay vỡ viền chữ.

> Quá trình chạy thực thi thực tế sẽ được tích hợp sẵn thông qua file `main_mobile.py`. Tốc độ kỳ vọng đạt tính bằng giây tùy theo sức mạnh GPU M-Series hoặc NPU/CPU của Mobile.
