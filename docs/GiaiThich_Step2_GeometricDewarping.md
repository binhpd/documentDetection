# Giải Thích Chi Tiết: STEP 2 - BIẾN ĐỔI HÌNH HỌC VÀ LÀM PHẲNG (Geometric & Dewarping)

Bước cực kỳ quan trọng đòi hỏi kiến thức Toán Đại số tuyến tính tĩnh (Linear Algebra) và Mạng lưới tọa độ Động (Neural Grids). Bước này sẽ nắn bóp tấm giấy bị méo lệch thành một bề mặt thẳng thớm chuẩn bị giao cho bước tăng cường đọc chữ.

---

## 1. Thuật toán Xếp Góc Cực Trị (Corner Sorting)
Tuy Bước 1 tìm được 4 tọa độ, nhưng bộ mảng đó lại vô trật tự (hoặc theo chiều kim đồng hồ ngẫu nhiên khởi điểm). Máy tính sẽ không biết đâu là "Góc trên Trái", đâu là "Góc dưới Phải". Nếu Warp nhầm hệ, tài liệu sẽ bị lật ngược.

* **Thuật toán sử dụng:** Kỹ thuật Sắp xếp Lưới Hình Học thông qua Phép Tính Tọa Độ (Cộng/Trừ 2 trục X,Y). Tham khảo module `corner_sorter.py`.
* **Quy trình Phân tích:**
  - `Tổng $S = X + Y$`: Pixel Góc Trên-Trái (Top-Left) sẽ gần tâm tọa độ máy tính nhất $\rightarrow S$ nhỏ nhất. Góc Dưới-Phải (Bottom-Right) xa nhất $\rightarrow S$ lớn nhất.
  - `Hiệu $D = Y - X$`: Pixel Góc Phải-Trên (Top-Right) có trục X cực to mà trục Y cực nhỏ $\rightarrow D = nhỏ nhất$. Góc Trái-Dưới (Bottom-Left) có Y to X nhỏ $\rightarrow D = lớn nhất$.
* **Input:** Biến mảng ngẫu nhiên 4 điểm.
* **Output:** Mã 1 chiều tuần tự 4 đỉnh tiêu chuẩn: `Top-Left`, `Top-Right`, `Bottom-Right`, `Bottom-Left`.

---

## 2. Thuật toán Chống Véo Biến Dạng (IoU Anti-Pinch)
Một công nghệ tự động chẩn đoán thông minh do nhóm xây dựng để ngăn Mạng UVDoc làm rách tài liệu dạng phẳng hình hộp. 

* **Toán học cốt lõi:** _Intersection over Union (IoU)_ – Tỷ lệ Tương Lập.
* **Cơ chế thực thi:**
  1. Vẽ một mặt nạ Đa giác 4 đường thẳng lý tưởng (Ideal Polygon) đi qua 4 điểm góc của `cv2.minAreaRect` (dùng hàm `cv2.fillPoly`). Gọi đây là **M_poly**.
  2. Lấy mask cong hình uốn lượn do U²-Net bóc tách thực tế ở Bước 1. Gọi là **M_real**.
  3. Lập phép đếm giao nhau (Intersection - Phép toán AND Boolean `bitise_and(M_poly, M_real)`) so với Tổng Phủ (Union - `bitwise_or`).
  4. Nếu $\text{IoU} > 0.94$ (94%): Chứng tỏ cái viền cắt thực tế của tờ giấy với đường biên tứ giác kẻ cứng lý tưởng gần như là MỘT. $\rightarrow$ Tính bằng **Phẳng tuyệt nhiên**. Lập tức Bác bỏ quyền chạy UVDoc, ép phải ném vào ma trận *Perspective*.
  5. Nếu $\text{IoU} < 0.94$: Viền thực tế hụt quá nhiều so với viền chữ nhật, tức khung tài liệu ngoài đời thực đã cong vênh gập bẻ. Kích hoạt *UVDoc*.

---

## 3. Thuật toán Căng Lưới Tương Phối Phẳng (Perspective Transform)
Đây là phương pháp Biến đổi Tọa độ truyền thống nếu tờ giấy được bắt xác định là 1 bề mặt 2D xoay trong một chiều 3D cứng (Không nhàu).

* **Bước A: Chốt Kính cỡ Cực Đại (Euclidean Distance)**
  - Tính khoảng cách lề ngang chiều dài (`L2_Norm` giữa `TL-TR` và `BL-BR`). Lấy cạnh lớn hơn làm Width.
  - Tính khoảng cách lề đứng (`L2_Norm` giữa `TL-BL` và `TR-BR`). Lấy cạnh dài hơn làm Height.
  - Việc này đảm bảo điểm ảnh sau khi bị lôi kéo ra không bị rỗ méo.
* **Bước B: Giải Ma Trận 3x3 Chéo**
  - Thuật toán DLT (Direct Linear Transformation - thực thi qua `cv2.getPerspectiveTransform`).
  - Ma trận 3x3 $M$ ánh xạ $Tứ_Giác_Méo \rightarrow Hình_Chữ_Nhật_Thẳng_{Width \times Height}$.
* **Bước C: Nội Suy Phối Cảnh (Interpolation Warp)**
  - Hàm `cv2.warpPerspective` nhân từng pixel của hình ảnh thô qua Ma Trận $M$. Các hàng chữ xéo góc sẽ dần bị bẻ ép căng góc lại vuông vức như bản scan quét phẳng lý tưởng. 

---

## 4. Dewarping Spline Võng Dòng Chữ (Coons Patch / UVDoc Neural Grid)

Đối với những trang sách dày, có gáy sách cong uốn lượn, phép biến đổi Ma trận ngang dọc Perspective chết đứng vì bẻ méo cấu trúc chữ.

### Hệ thống UVDoc (ML 3D Grid Warp)
* **Ý nghĩa:** Giải cấu trúc giấy gấp nếp cong vút về một ma trận tọa độ vector đảo chiều 2 chiều (UV Map).
* **Input:** Một bức ảnh lọt thỏm trong U2-Net mask.
* **Quy trình:**
  1. Mô hình dự đoán ma trận quang sai độ lõm của vật lý ánh sáng hắt lại ống kính. 
  2. Map đó sinh ra bản đồ uốn (Deformation Field Map), ví dụ điểm pixel A lỡ trôi lên 12px vì uốn cong sẽ bị kéo chùng lại 12px dọc chiều Y để thẳng góc với mắt.
* **Output:** Xóa nếp gấp gáy sách, văn bản thẳng tắp dọc hàng không bị đu đưa vẹo.
