# Xây dựng Hệ thống Tự động Căn chỉnh và Làm rõ nét Ảnh chụp Tài liệu

> **Môn học:** Xử lý Ảnh và Video
> **Nhóm:** 6

---

## 1. Giới thiệu bài toán

Trong thực tế, người dùng thường chụp tài liệu (hợp đồng, hóa đơn, bài giảng, sách…) bằng camera điện thoại. Ảnh thu được thường gặp các vấn đề:

- **Nghiêng / méo phối cảnh** do góc chụp không vuông góc với mặt phẳng tài liệu.
- **Ánh sáng không đều** – một phần tài liệu bị tối, phần khác bị lóa hoặc có bóng đổ.
- **Nhiễu cảm biến** và **mờ nét** do rung tay hoặc điều kiện chụp thiếu sáng.

**Mục tiêu** của hệ thống là biến một bức ảnh chụp từ điện thoại (bị nghiêng, nhiễu, ánh sáng không đều) thành một bản scan phẳng và rõ nét — tương tự chức năng của các ứng dụng Scanner phổ biến (CamScanner, Adobe Scan, Microsoft Lens…).

---

## 2. Pipeline xử lý tổng quan

```
Ảnh đầu vào (chụp từ điện thoại)
        │
        ▼
┌───────────────────────────────────┐
│  Bước 1: Hiệu chỉnh Hình học     │
│  & Phối cảnh                      │
│  ─────────────────────────────    │
│  • Phát hiện vùng tài liệu       │
│  • Biến đổi Perspective Warp     │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  Bước 2: Xử lý Ánh sáng          │
│  & Tương phản                     │
│  ─────────────────────────────    │
│  • Adaptive Thresholding          │
│  • Morphological Operations       │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  Bước 3: Tăng cường Chi tiết      │
│  & Khử nhiễu                      │
│  ─────────────────────────────    │
│  • Unsharp Masking / High-pass    │
│  • Edge-preserving Denoising      │
└───────────────┬───────────────────┘
                │
                ▼
        Ảnh đầu ra (bản scan)
```

---

## 3. Chi tiết từng bước xử lý

### 3.1. Hiệu chỉnh Hình học và Phối cảnh (Geometric & Perspective Correction)

#### 3.1.1. Phát hiện vùng chứa tài liệu (Document Localization)

**Mục đích:** Xác định bốn đỉnh (hoặc đa giác lồi) bao quanh tài liệu trong ảnh, tách tài liệu ra khỏi nền.

**Thuật toán:**

1. **Tiền xử lý:**
   - Chuyển ảnh sang Grayscale.
   - Áp dụng bộ lọc Gaussian Blur để giảm nhiễu, tránh phát hiện cạnh giả.

2. **Phát hiện cạnh (Edge Detection):**
   - Sử dụng thuật toán **Canny Edge Detector** để trích xuất các cạnh trong ảnh.
   - Canny sử dụng hai ngưỡng (low threshold, high threshold) kết hợp Non-Maximum Suppression để cho ra cạnh mảnh và chính xác.

3. **Tìm contour và xấp xỉ đa giác lồi:**
   - Tìm tất cả các contour từ ảnh cạnh bằng `cv2.findContours()`.
   - Sắp xếp contour theo diện tích giảm dần, chọn contour lớn nhất có khả năng là tài liệu.
   - Xấp xỉ contour thành đa giác bằng thuật toán **Douglas-Peucker** (`cv2.approxPolyDP()`).
   - Nếu đa giác có đúng **4 đỉnh** → đó là vùng tài liệu (hình tứ giác).

**Công thức xấp xỉ đa giác (Douglas-Peucker):**

$$\epsilon = \alpha \times \text{arcLength}(contour)$$

Trong đó $\alpha$ thường nằm trong khoảng $[0.02, 0.05]$.

#### 3.1.2. Biến đổi Perspective Warp

**Mục đích:** Ánh xạ vùng tài liệu (tứ giác bất kỳ) về hình chữ nhật chuẩn, loại bỏ méo phối cảnh.

**Thuật toán:**

1. **Sắp xếp 4 đỉnh** theo thứ tự: top-left, top-right, bottom-right, bottom-left.
2. **Tính kích thước ảnh đầu ra** dựa trên khoảng cách Euclidean giữa các đỉnh:

$$W = \max\left(\|P_{TR} - P_{TL}\|, \|P_{BR} - P_{BL}\|\right)$$

$$H = \max\left(\|P_{TL} - P_{BL}\|, \|P_{TR} - P_{BR}\|\right)$$

3. **Tính ma trận biến đổi phối cảnh** $M$ (3×3):

$$M = \texttt{cv2.getPerspectiveTransform}(\text{src\_pts}, \text{dst\_pts})$$

   Trong đó `src_pts` là 4 đỉnh tài liệu, `dst_pts` là 4 đỉnh hình chữ nhật đích.

4. **Áp dụng biến đổi:**

$$I_{out} = \texttt{cv2.warpPerspective}(I_{in}, M, (W, H))$$

**Kết quả:** Ảnh tài liệu được "duỗi phẳng" về góc nhìn trực diện 90°, loại bỏ hoàn toàn sai số do góc chụp nghiêng.

---

### 3.2. Xử lý Ánh sáng và Tương phản (Illumination & Contrast Optimization)

#### 3.2.1. Adaptive Thresholding (Nhị phân hóa thích nghi)

**Mục đích:** Xử lý tình trạng ánh sáng không đồng nhất — khi một phần ảnh sáng và phần khác tối, ngưỡng cố định (global threshold) không thể phân tách tốt chữ và nền.

**Nguyên lý:**

Thay vì dùng một giá trị ngưỡng $T$ duy nhất cho toàn bộ ảnh, Adaptive Thresholding tính ngưỡng $T(x, y)$ **riêng cho từng pixel** dựa trên giá trị trung bình (hoặc trung bình có trọng số Gaussian) của vùng lân cận kích thước $B \times B$:

$$T(x, y) = \mu_{B}(x, y) - C$$

Trong đó:
- $\mu_{B}(x, y)$: giá trị trung bình (mean) hoặc trung bình Gaussian của vùng lân cận $B \times B$ quanh pixel $(x, y)$.
- $C$: hằng số điều chỉnh (thường $C \in [2, 15]$).

**Quy tắc nhị phân hóa:**

$$I_{out}(x, y) = \begin{cases} 255 & \text{nếu } I(x, y) > T(x, y) \\ 0 & \text{ngược lại} \end{cases}$$

**Hai phương pháp phổ biến:**

| Phương pháp | Cách tính $\mu_B$ | Đặc điểm |
|---|---|---|
| `ADAPTIVE_THRESH_MEAN_C` | Trung bình cộng | Đơn giản, nhanh |
| `ADAPTIVE_THRESH_GAUSSIAN_C` | Trung bình có trọng số Gaussian | Cho trọng số cao hơn ở pixel gần tâm, kết quả mượt hơn |

#### 3.2.2. Morphological Operations (Phép toán hình thái học)

**Mục đích:** Loại bỏ bóng đổ, làm sạch nền văn bản, loại bỏ nhiễu nhỏ sau bước nhị phân hóa.

**Các phép toán chính:**

| Phép toán | Công thức | Tác dụng |
|---|---|---|
| **Erosion** (Co) | $A \ominus B = \{z \mid B_z \subseteq A\}$ | Thu nhỏ vùng trắng, loại bỏ nhiễu nhỏ |
| **Dilation** (Giãn) | $A \oplus B = \{z \mid B_z \cap A \neq \emptyset\}$ | Mở rộng vùng trắng, nối các vùng bị đứt |
| **Opening** (Mở) | $A \circ B = (A \ominus B) \oplus B$ | Loại bỏ nhiễu nhỏ, giữ nguyên hình dạng chính |
| **Closing** (Đóng) | $A \bullet B = (A \oplus B) \ominus B$ | Lấp đầy lỗ nhỏ, nối các vùng gần nhau |

Trong đó $B$ là **phần tử cấu trúc** (structuring element), thường là hình vuông, hình tròn hoặc hình chữ thập.

**Quy trình loại bỏ bóng đổ:**

1. Áp dụng **Dilation** với kernel lớn để tạo ảnh nền ước lượng (background estimation).
2. Trừ ảnh gốc cho ảnh nền: $I_{no\_shadow} = I_{dilated} - I_{original}$.
3. Chuẩn hóa và đảo ngược để thu được ảnh sạch bóng.

---

### 3.3. Tăng cường Chi tiết và Khử nhiễu (Detail Enhancement & Denoising)

#### 3.3.1. Làm sắc nét (Sharpening)

**Mục đích:** Tăng cường các nét chữ bị mờ do rung tay hoặc lấy nét sai.

**Phương pháp 1: Unsharp Masking**

Nguyên lý: Tăng cường chi tiết bằng cách cộng thêm phần "chi tiết cao tần" vào ảnh gốc.

$$I_{sharp} = I + \alpha \cdot (I - I_{blur})$$

Trong đó:
- $I$: ảnh gốc.
- $I_{blur}$: ảnh sau khi làm mờ Gaussian (thành phần tần số thấp).
- $(I - I_{blur})$: thành phần tần số cao (chi tiết, cạnh).
- $\alpha$: hệ số tăng cường ($\alpha > 0$, thường $\alpha \in [0.5, 2.0]$).

**Phương pháp 2: High-pass Filter (Bộ lọc thông cao)**

Sử dụng kernel tích chập để trích xuất và tăng cường cạnh:

$$K_{highpass} = \begin{bmatrix} -1 & -1 & -1 \\ -1 & 9 & -1 \\ -1 & -1 & -1 \end{bmatrix}$$

Kernel này tương đương với: $K = I_{identity} + \text{Laplacian}$, vừa giữ ảnh gốc vừa tăng cường cạnh.

#### 3.3.2. Khử nhiễu bảo toàn biên (Edge-preserving Denoising)

**Mục đích:** Giảm nhiễu cảm biến (sensor noise) mà không làm mờ các cạnh quan trọng (nét chữ, đường kẻ).

**Phương pháp 1: Bilateral Filter**

$$I_{out}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) \cdot f_r(\|I(x_i) - I(x)\|) \cdot g_s(\|x_i - x\|)$$

Trong đó:
- $g_s$: kernel Gaussian theo **khoảng cách không gian** (spatial) — pixel gần được ưu tiên hơn.
- $f_r$: kernel Gaussian theo **cường độ sáng** (range) — pixel có cường độ tương tự mới được trung bình hóa.
- $W_p$: hệ số chuẩn hóa.

Bilateral Filter khử nhiễu hiệu quả tại vùng phẳng nhưng **bảo toàn biên** vì pixel khác biệt cường độ lớn (tại biên) sẽ có trọng số $f_r$ rất nhỏ.

**Phương pháp 2: Non-Local Means Denoising (NLM)**

$$I_{out}(x) = \frac{1}{C(x)} \sum_{y \in V(x)} w(x, y) \cdot I(y)$$

Trong đó trọng số $w(x, y)$ được tính dựa trên **độ tương đồng giữa patch** (vùng nhỏ) xung quanh pixel $x$ và pixel $y$:

$$w(x, y) = \exp\left(-\frac{\|P(x) - P(y)\|^2}{h^2}\right)$$

- $P(x)$: patch xung quanh pixel $x$.
- $h$: tham số điều khiển mức độ lọc.

NLM cho kết quả khử nhiễu tốt hơn Bilateral Filter nhờ so sánh cấu trúc patch thay vì chỉ so sánh cường độ từng pixel.

---

## 4. Tóm tắt các thuật toán và thư viện sử dụng

| Bước | Thuật toán | Hàm OpenCV |
|---|---|---|
| Phát hiện cạnh | Canny Edge Detection | `cv2.Canny()` |
| Tìm contour | Suzuki-Abe | `cv2.findContours()` |
| Xấp xỉ đa giác | Douglas-Peucker | `cv2.approxPolyDP()` |
| Biến đổi phối cảnh | Perspective Transform | `cv2.getPerspectiveTransform()` + `cv2.warpPerspective()` |
| Nhị phân hóa thích nghi | Adaptive Thresholding | `cv2.adaptiveThreshold()` |
| Hình thái học | Erosion, Dilation, Opening, Closing | `cv2.erode()`, `cv2.dilate()`, `cv2.morphologyEx()` |
| Làm sắc nét | Unsharp Masking | `cv2.GaussianBlur()` + phép trừ/cộng |
| Bộ lọc thông cao | Convolution | `cv2.filter2D()` |
| Khử nhiễu Bilateral | Bilateral Filter | `cv2.bilateralFilter()` |
| Khử nhiễu NLM | Non-Local Means | `cv2.fastNlMeansDenoising()` |

---

## 5. Công nghệ và Công cụ

- **Ngôn ngữ:** Python 3.x
- **Thư viện chính:** OpenCV (`cv2`), NumPy
- **Thư viện hỗ trợ:** Matplotlib (hiển thị kết quả), imutils (tiện ích xử lý ảnh)
- **Môi trường:** Jupyter Notebook / Google Colab

---

## 6. Dữ liệu đầu vào / đầu ra

### Đầu vào
- Ảnh chụp tài liệu từ camera điện thoại (định dạng JPG/PNG).
- Ảnh có thể bị: nghiêng, méo phối cảnh, ánh sáng không đều, nhiễu, mờ.

### Đầu ra
- Ảnh tài liệu đã được:
  - Căn chỉnh phẳng (perspective corrected).
  - Tăng tương phản, loại bỏ bóng đổ.
  - Làm rõ nét chữ, khử nhiễu.
- Tương đương chất lượng bản scan từ máy scanner.

---

## 7. Tiêu chí đánh giá

| Tiêu chí | Mô tả |
|---|---|
| **Độ chính xác phát hiện tài liệu** | Tỷ lệ phát hiện đúng 4 góc tài liệu trên tập ảnh thử nghiệm |
| **Chất lượng căn chỉnh** | Mức độ "phẳng" của ảnh sau biến đổi phối cảnh (đánh giá trực quan) |
| **Chất lượng ảnh đầu ra** | Độ rõ nét chữ, mức độ loại bỏ nhiễu và bóng đổ (PSNR, SSIM) |
| **Tốc độ xử lý** | Thời gian xử lý trung bình trên mỗi ảnh |

---

## 8. Tài liệu tham khảo

1. Bradski, G., & Kaehler, A. (2008). *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media.
2. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th Edition). Pearson.
3. OpenCV Documentation: https://docs.opencv.org/
4. Buades, A., Coll, B., & Morel, J.-M. (2005). A non-local algorithm for image denoising. *CVPR 2005*.
5. Tomasi, C., & Manduchi, R. (1998). Bilateral Filtering for Gray and Color Images. *ICCV 1998*.
