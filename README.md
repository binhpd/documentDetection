# Dự án Quét và Phân tích Tài liệu (Document Scanner)

Dự án này bao gồm hai cách tiếp cận chính để xử lý trích xuất và tối ưu hoá giao diện tài liệu (Document Scanner/Dewarping):

1. **Pipeline Without ML (Truyền thống)**: Dựa hoàn toàn vào các kỹ thuật xử lý ảnh (Computer Vision) thuần tuý thông qua OpenCV như Canny Edge Detection, Hough Lines, Thresholding, Morphology.
2. **Pipeline With ML (Hiện đại)**: Kết hợp các kỹ thuật Computer Vision với sức mạnh của Trí tuệ Nhân tạo (Machine Learning/Deep Learning) sử dụng các mô hình học sâu như **DocAligner**, **YOLOv8** để định vị góc cực kỳ chính xác.

---

## 1. Pipeline Wihtout ML (Chỉ dùng Xử lý ảnh OpenCV)

Thư mục: `Pipeline Without ML/`

Đây là cách tiếp cận ban đầu, sử dụng các phép biến đổi toán học phân đoạn. Gồm các bước rời rạc để bạn nghiên cứu từng phần:

* **Step 1 canny edge detection**:
    * Phát hiện cạnh bằng thuật toán Canny.
    * Tìm Contour lớn nhất có 4 đỉnh (`approxPolyDP`) hoặc nội suy bằng Hough Lines nếu bị khuyết.
    * *Lệnh chạy (Ví dụ)*: `python3 "Pipeline Without ML/Step 1 canny edge detection/main.py"`

* **Step 2 perspective transform**:
    * Cắt ảnh và biến đổi phối cảnh 3D thành bản phẳng 2D dựa trên 4 góc đã tìm được ở Step 1.
    * *Lệnh chạy (Ví dụ)*: `python3 "Pipeline Without ML/Step 2 perspective transform/main.py"`

* **Step 3 enhancement**:
    * Khử đổ bóng (Shadow Removal) bằng phép chia ảnh với nền (Dilate & Gaussian).
    * Binarization (nhị phân hoá chữ đen nền trắng) để thành chất lượng chuẩn máy scan.
    * *Lệnh chạy (Ví dụ)*: `python3 "Pipeline Without ML/Step 3 enhancement/main.py"`

> **Lưu ý**: Nhược điểm của phương pháp này là dễ bị thất bại với các ảnh tối, nền phức tạp hoặc tài liệu bị ngón tay che khuất viền.

---

## 2. Pipeline With ML (Sử dụng AI Phân vùng vùng ảnh)

Thư mục: `Pipeline With ML/`

Đây là Pipeline rút gọn, nối tự động (End-to-End) cả 3 quá trình trên vào trong một file chạy duy nhất. Đặc biệt, nó tích hợp **Mạng nơ-ron Tích chập (CNN)** chuyên dụng giải quyết mọi nhược điểm của OpenCV truyền thống.

### Hướng dẫn Cài đặt & Khởi chạy (Môi trường ML):
Pipeline này sử dụng Machine Learning nên yêu cầu cài đặt môi trường ảo riêng biệt chứa các bộ thư viện hạng nặng (`torch`, `onnxruntime`, `docaligner`...).

**Bước 1: Tạo môi trường ảo (Tuỳ chọn nhưng Khuyến nghị)**
```bash
# Tạo môi trường ảo tên là venv2
python3 -m venv venv2
```

**Bước 2: Kích hoạt môi trường ảo**
```bash
# Trên macOS/Linux:
source venv2/bin/activate
# Trên Windows:
# venv2\Scripts\activate
```

**Bước 3: Cài đặt các thư viện cơ bản**
```bash
pip install -r requirements.txt
```

**Bước 4: Cài đặt thư viện ML DocAligner (Bắt buộc cho tuỳ chọn --docaligner)**
DocAligner cần được cài từ file wheel của tác giả thay vì qua pip thông thường:
```bash
pip install https://github.com/DocsaidLab/DocAligner/releases/download/v0.1.0/docaligner-0.1.0-py3-none-any.whl
```

### Các Lệnh Chạy Toàn Diện:

**1. Chạy với ảnh tự chọn bất kỳ:**
```bash
python3 "Pipeline With ML/main.py" "/đường/dẫn/đến/file/ảnh.jpg"
```

**2. Sử dụng siêu mô hình AI DocAligner (Độ chính xác cao nhất):**
AI tự động vẽ bản đồ nhiệt và tìm chính xác 4 đỉnh tài liệu dù nền rất hỗn loạn hay móp méo.
```bash
python3 "Pipeline With ML/main.py" <thư_mục_test> <số_thứ_tự> --docaligner

# Ví dụ test trên ảnh chụp nghiêng số 9:
python3 "Pipeline With ML/main.py" perspective 9 --docaligner
```

**3. Liệt kê các bộ ảnh có sẵn trong hệ thống:**
```bash
python3 "Pipeline With ML/main.py" list
```

**4. Kích hoạt ML Dewarping lật phẳng giấy cong (Chưa bao gồm model):**
```bash
python3 "Pipeline With ML/main.py" <thư_mục_test> <số_thứ_tự> --dewarp-ml
```

---

## 3. Tài liệu Báo cáo môn học

Các tài liệu lý thuyết, sơ đồ khối và đánh giá ưu/nhược điểm được lưu trữ trong thư mục `docs/`:

- [Mô tả Bài toán (Lý thuyết Image Processing cơ bản)](docs/BaiToan.md)
- [Đánh giá Giải pháp (Phân tích các ca khó & Case Limit)](docs/DanhGiaGiaiPhap.md)
- [Workflow Chi tiết (Từng bước của Pipeline có ML)](docs/PIPELINE_WORKFLOW.md)
