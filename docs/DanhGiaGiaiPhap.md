# Đánh giá Giải pháp Hiện tại với các Trường hợp Ảnh Cực đoan

> **Nhóm:** 6 — Hệ thống Tự động Căn chỉnh và Làm rõ nét Ảnh chụp Tài liệu
> **Dataset:** 7 categories × 170 ảnh = **1.020 ảnh**

---

## 1. Tổng quan Giải pháp Hiện tại

### Pipeline hiện tại (3 bước)

```
Ảnh gốc
  │
  ▼  [Step 1] Canny Edge → findContours → approxPolyDP (tìm đúng 4 đỉnh)
  │
  ▼  [Step 2] getPerspectiveTransform → warpPerspective (4 điểm → hình chữ nhật)
  │
  ▼  [Step 3] Shadow Removal → Adaptive Threshold → Morphology
  │
  ▼
Ảnh scan phẳng
```

### 3 giả định cốt lõi (assumptions) của pipeline

| # | Giả định | Mô tả |
|---|----------|-------|
| **A1** | Tài liệu có **4 cạnh thẳng** | `approxPolyDP` tìm polygon 4 đỉnh → yêu cầu 4 cạnh thẳng rõ ràng |
| **A2** | **Toàn bộ 4 góc** đều nằm trong khung hình | Cần đủ 4 điểm cho `getPerspectiveTransform` |
| **A3** | Bề mặt tài liệu là **mặt phẳng** | `warpPerspective` là phép biến đổi tuyến tính, chỉ đúng với mặt phẳng |

---

## 2. Phân tích từng Loại Ảnh Cực đoan

### 2.1. Curved (Trang bị cong/cuộn)

**Đặc điểm ảnh:**
- Trang giấy bị cong do cầm tay hoặc đặt trên bề mặt không phẳng
- Cạnh tài liệu là **đường cong**, không phải đường thẳng
- Chữ bị biến dạng theo đường cong bề mặt

**Giả định bị vi phạm:** **A1** (cạnh cong, không thẳng) + **A3** (bề mặt không phẳng)

**Vấn đề cụ thể với code hiện tại:**

```python
# corner_detector.py — tìm polygon 4 đỉnh
approx = cv2.approxPolyDP(c, self.approx_epsilon * peri, True)
if len(approx) == 4:      # ← THẤT BẠI: contour cong → xấp xỉ > 4 đỉnh
    return approx
```

| Bước | Trạng thái | Lý do |
|------|-----------|-------|
| Canny Edge | ⚠️ Hoạt động kém | Cạnh cong tạo gradient yếu hơn cạnh thẳng |
| findContours | ⚠️ Tìm được nhưng cong | Contour theo đường cong, không phải tứ giác |
| approxPolyDP = 4 | ❌ **THẤT BẠI** | Đường cong cần > 4 điểm để xấp xỉ chính xác |
| warpPerspective | ❌ Sai hoàn toàn | Ngay cả khi ép 4 điểm, chữ vẫn cong vì bề mặt không phẳng |

**Mức độ thất bại: NGHIÊM TRỌNG — Pipeline không thể xử lý.**

**Giải pháp đề xuất:**

| Approach | Phương pháp | Độ phức tạp | Hiệu quả |
|----------|-------------|-------------|-----------|
| **B1: Page Dewarping** | Phát hiện text lines → ước lượng bề mặt 3D → biến đổi ngược (unwarp) | Cao | Tốt |
| **B2: Thin-Plate Spline (TPS)** | Dùng nhiều điểm kiểm soát (> 4) → biến đổi phi tuyến | Trung bình | Khá |
| **B3: Mesh Warping** | Chia ảnh thành lưới → warp từng ô theo đường cong cục bộ | Cao | Rất tốt |

Thuật toán cụ thể cho B1 — Page Dewarping:
1. Phát hiện các dòng chữ (text line detection) bằng horizontal projection hoặc connected components
2. Mỗi dòng chữ lý tưởng là đường thẳng ngang → dòng cong = bề mặt cong
3. Fit cubic spline cho mỗi dòng → ước lượng hàm biến dạng $f(x, y)$
4. Áp dụng biến đổi ngược để "duỗi" các dòng thành đường thẳng

---

### 2.2. Fold (Gấp/Nhăn)

**Đặc điểm ảnh:**
- Tài liệu có vết gấp (fold lines) tạo ra các mặt phẳng khác nhau
- Bóng đổ dọc theo vết gấp
- Một số góc bị gấp vào → contour không còn hình tứ giác

**Giả định bị vi phạm:** **A3** (nhiều mặt phẳng khác nhau) + phần nào **A1** (góc bị gấp)

**Vấn đề cụ thể:**

| Bước | Trạng thái | Lý do |
|------|-----------|-------|
| Canny Edge | ✅ Hoạt động | Cạnh vẫn tạo gradient mạnh |
| findContours | ⚠️ Tìm được nhưng méo | Vết gấp tạo contour phụ, contour ngoài bị biến dạng |
| approxPolyDP = 4 | ⚠️ Có thể tìm 4 điểm nhưng sai vị trí | Vết gấp đẩy contour lệch khỏi góc thật |
| warpPerspective | ⚠️ Warp được nhưng vẫn nhăn | Chỉ chỉnh phối cảnh tổng thể, không sửa được nếp gấp cục bộ |
| Shadow removal | ⚠️ Xử lý kém | Bóng đổ dọc vết gấp rất đậm, kernel dilation không đủ lớn |

**Mức độ thất bại: TRUNG BÌNH — Pipeline hoạt động một phần nhưng kết quả kém.**

**Giải pháp đề xuất:**

| Approach | Phương pháp | Mô tả |
|----------|-------------|-------|
| **F1: Fold line detection** | Hough Line Transform trên ảnh gradient | Phát hiện vết gấp → chia ảnh thành vùng → warp riêng từng vùng |
| **F2: Piecewise Perspective** | Chia tài liệu thành sub-regions | Mỗi vùng giữa 2 vết gấp là 1 mặt phẳng → warp riêng → ghép lại |
| **F3: Shadow-aware enhancement** | Tăng cường shadow removal | Dùng CLAHE (Contrast Limited Adaptive Histogram Equalization) + large kernel morphology |

---

### 2.3. Incomplete (Thiếu góc / Gấp mép / Bị che)

**Đặc điểm ảnh:**
- Một hoặc nhiều góc tài liệu bị cắt khỏi khung hình
- Ngón tay/bàn tay che một phần tài liệu
- Góc tài liệu bị gấp vào trong

**Giả định bị vi phạm:** **A2** (không đủ 4 góc trong khung hình)

**Vấn đề cụ thể:**

```python
# corner_detector.py
if len(approx) == 4:       # ← Nếu 1 góc bị che → có thể tìm được 3 hoặc 5+ đỉnh
    return approx
return None                 # ← Trả về None → TOÀN BỘ pipeline dừng lại
```

| Bước | Trạng thái | Lý do |
|------|-----------|-------|
| Canny Edge | ✅ Hoạt động | Phần cạnh còn lại vẫn phát hiện được |
| findContours | ⚠️ Contour không khép kín | Thiếu góc → contour bị hở hoặc bao gồm cả tay |
| approxPolyDP = 4 | ❌ **THẤT BẠI** | Không đủ 4 đỉnh, hoặc tìm sai đỉnh (bao gồm ngón tay) |
| getPerspectiveTransform | ❌ Không thực hiện được | Cần chính xác 4 điểm nguồn |

**Mức độ thất bại: NGHIÊM TRỌNG — Pipeline dừng ngay bước phát hiện góc.**

**Giải pháp đề xuất:**

| Approach | Phương pháp | Mô tả |
|----------|-------------|-------|
| **I1: Hough Line extrapolation** | Tìm đường thẳng → kéo dài → tìm giao điểm | Phát hiện 2 cạnh thấy được → kéo dài → suy ra góc bị thiếu |
| **I2: Convex Hull + fallback** | Nới lỏng điều kiện 4 đỉnh | Dùng convex hull thay vì approxPolyDP chính xác |
| **I3: Heuristic corner estimation** | Dùng 3 góc + tỷ lệ A4 | Từ 3 góc tìm được + giả sử tỷ lệ A4 (√2:1) → suy ra góc thứ 4 |
| **I4: Semantic segmentation** | Deep Learning (U-Net) | Phân vùng pixel-level: tài liệu vs nền vs tay → robust hơn contour |

Thuật toán cụ thể cho I1 — Hough Line Extrapolation:
1. Phát hiện cạnh bằng Canny
2. Dùng `cv2.HoughLinesP()` để tìm các đoạn thẳng
3. Gom nhóm (cluster) các đoạn thẳng gần nhau thành 4 đường thẳng chính
4. Kéo dài mỗi đường thẳng ra ngoài khung hình
5. Tìm 4 giao điểm → đó là 4 góc tài liệu (kể cả khi góc nằm ngoài ảnh)

---

### 2.4. Perspective (Góc chụp cực đoan)

**Đặc điểm ảnh:**
- Tài liệu chụp từ góc rất nghiêng (> 45°)
- Tài liệu nhỏ trong khung hình, nền phức tạp (cỏ, lá, đất)
- Có vật thể che phủ (cành cây, lá)

**Giả định bị vi phạm:** Không vi phạm trực tiếp, nhưng **chất lượng kém do tham số cứng**

**Vấn đề cụ thể:**

| Bước | Trạng thái | Lý do |
|------|-----------|-------|
| Canny (75, 200) | ❌ **THẤT BẠI** | Nền cỏ/đất tạo RẤT NHIỀU cạnh giả, tràn ngập cạnh tài liệu |
| GaussianBlur (5,5) | ⚠️ Không đủ | Kernel (5,5) quá nhỏ cho ảnh nhiều chi tiết nền |
| findContours | ❌ Contour sai | Contour lớn nhất có thể là đám cỏ, không phải tài liệu |
| approxPolyDP | ❌ Tìm sai object | Xấp xỉ polygon trên contour sai |

**Mức độ thất bại: CAO — Canny thất bại do nền phức tạp.**

**Giải pháp đề xuất:**

| Approach | Phương pháp | Mô tả |
|----------|-------------|-------|
| **P1: Auto Canny threshold** | Tự động tính ngưỡng từ median | `threshold = median ± 0.33 * median` — thích nghi với mỗi ảnh |
| **P2: Color-based segmentation** | Phân tách theo màu giấy | Giấy trắng nổi bật trên nền cỏ xanh → segmentation bằng color space (HSV/LAB) |
| **P3: Larger Gaussian + Dilate** | Tăng preprocessing | Blur kernel lớn hơn (15,15) + dilate cạnh để nối → contour liền mạch hơn |
| **P4: GrabCut** | Semi-auto segmentation | `cv2.grabCut()` với bounding box ước lượng → tách nền chính xác hơn |

---

### 2.5. Rotate (Xoay mạnh)

**Đặc điểm ảnh:**
- Tài liệu xoay 30°-90° so với trục ngang
- Có bàn tay cầm giữ
- Phần cạnh bị tay che

**Giả định bị vi phạm:** Phần nào **A2** (tay che góc) + tham số Canny không tối ưu

**Vấn đề cụ thể:**

| Bước | Trạng thái | Lý do |
|------|-----------|-------|
| Canny Edge | ✅ Hoạt động | Cạnh giấy tương phản tốt với nền |
| findContours | ⚠️ Contour bao gồm cả tay | Tay + giấy tạo thành 1 khối trắng → contour sai |
| approxPolyDP = 4 | ⚠️ Có thể 4 nhưng sai | 4 đỉnh bao gồm ngón tay thay vì góc giấy |
| warpPerspective | ⚠️ Warp nhưng bao gồm cả tay | Kết quả có phần tay lẫn vào |
| **Text orientation** | ❌ Không xử lý | Chữ bị xoay 90° → cần phát hiện hướng và xoay lại |

**Mức độ thất bại: TRUNG BÌNH — Hoạt động nếu tay không che góc, nhưng thiếu text orientation correction.**

**Giải pháp đề xuất:**

| Approach | Phương pháp | Mô tả |
|----------|-------------|-------|
| **R1: Skin color filtering** | Loại vùng da trước khi tìm contour | Chuyển HSV → mask vùng da → loại khỏi edge map |
| **R2: Text orientation detection** | Phát hiện hướng chữ | Dùng Hough Transform hoặc minAreaRect → tính góc xoay → `cv2.rotate()` |
| **R3: Tesseract OSD** | OCR-based orientation | `pytesseract.image_to_osd()` phát hiện orientation + script → xoay lại |

---

### 2.6. Random (Nhàu nát / Hỗn hợp)

**Đặc điểm ảnh:**
- Giấy bị vò nhàu, nhiều nếp nhăn
- Bề mặt biến dạng phức tạp, không theo quy luật
- Kết hợp nhiều vấn đề: nhăn + gấp + bóng đổ + perspective

**Giả định bị vi phạm:** **A1 + A2 + A3 — Tất cả 3 giả định đều bị vi phạm**

| Bước | Trạng thái | Lý do |
|------|-----------|-------|
| Toàn bộ pipeline | ❌ **THẤT BẠI HOÀN TOÀN** | Không có cạnh thẳng, không có 4 góc rõ ràng, bề mặt biến dạng phức tạp |

**Mức độ thất bại: CỰC KỲ NGHIÊM TRỌNG — Nằm ngoài khả năng của traditional CV.**

**Giải pháp đề xuất:**

| Approach | Phương pháp | Mô tả |
|----------|-------------|-------|
| **X1: Deep Learning segmentation** | U-Net / Mask R-CNN | Học pixel-level segmentation → không phụ thuộc cạnh thẳng |
| **X2: DocTr / DewarpNet** | Chuyên biệt dewarping | Mô hình deep learning chuyên cho document dewarping |
| **X3: Chấp nhận giới hạn** | Skip hoặc cảnh báo | Nhận diện ảnh "quá khó" và báo người dùng chụp lại |

---

## 3. Bảng Tổng hợp Đánh giá

| Loại ảnh | Số lượng | Step 1 (Detection) | Step 2 (Warp) | Step 3 (Enhance) | Đánh giá tổng |
|----------|----------|---------------------|---------------|-------------------|---------------|
| **Perspective** (nhẹ) | — | ✅ | ✅ | ✅ | ✅ Tốt |
| **Rotate** (không che tay) | — | ✅ | ✅ | ✅ | ⚠️ Thiếu text orientation |
| **Fold** (nhẹ) | — | ⚠️ | ⚠️ | ⚠️ | ⚠️ Kết quả trung bình |
| **Perspective** (cực đoan) | 170 | ❌ | ❌ | — | ❌ Canny fail do nền |
| **Rotate** (có tay che) | 170 | ❌ | ❌ | — | ❌ Contour bao gồm tay |
| **Incomplete** | 170 | ❌ | ❌ | — | ❌ Thiếu góc |
| **Curved** | 170 | ❌ | ❌ | — | ❌ Cạnh cong |
| **Fold** (nặng) | 170 | ⚠️ | ❌ | ⚠️ | ❌ Đa mặt phẳng |
| **Random** (nhàu) | 170 | ❌ | ❌ | ❌ | ❌ Ngoài khả năng |

**Ước tính tỷ lệ thành công trên toàn dataset: ~15-25%** (chỉ perspective nhẹ + rotate đơn giản)

---

## 4. Lộ trình Cải thiện (Đề xuất theo Priority)

### Phase 1: Quick Wins — Cải thiện lớn, effort thấp

| # | Cải tiến | Ảnh hưởng | Effort |
|---|---------|-----------|--------|
| 1.1 | **Auto Canny threshold** (median-based) | Perspective cải thiện nhiều | Thấp |
| 1.2 | **Nới lỏng approxPolyDP** (cho phép 4-6 đỉnh, chọn convex hull) | Fold, Curved cải thiện một phần | Thấp |
| 1.3 | **Hough Line Extrapolation** cho trường hợp thiếu góc | Incomplete giải quyết được | Trung bình |
| 1.4 | **Skin color filtering** loại vùng da | Rotate cải thiện nhiều | Thấp |

### Phase 2: Moderate — Giải quyết phần lớn edge cases

| # | Cải tiến | Ảnh hưởng | Effort |
|---|---------|-----------|--------|
| 2.1 | **Color-based document segmentation** (HSV/LAB) | Perspective (nền phức tạp) | Trung bình |
| 2.2 | **Text orientation detection** + auto-rotate | Rotate hoàn chỉnh | Trung bình |
| 2.3 | **CLAHE + enhanced shadow removal** | Fold bóng đổ | Thấp |
| 2.4 | **Piecewise perspective** cho fold | Fold trung bình | Cao |

### Phase 3: Advanced — Giải quyết các trường hợp cực đoan

| # | Cải tiến | Ảnh hưởng | Effort |
|---|---------|-----------|--------|
| 3.1 | **Page Dewarping** (text line based) | Curved giải quyết tốt | Rất cao |
| 3.2 | **U-Net document segmentation** | Tất cả categories | Rất cao (cần training data) |
| 3.3 | **DewarpNet / DocTr** (pretrained models) | Curved + Fold + Random | Cao (cần GPU) |

---

## 5. Giới hạn Cơ bản của Traditional Computer Vision

```
                        Độ phức tạp ảnh
                ──────────────────────────────►
  Traditional   │  Perspective  │  Rotate     │  Fold     │  Curved   │  Random
  CV (hiện tại) │  ✅ Tốt       │  ⚠️ OK      │  ⚠️ Kém   │  ❌ Fail  │  ❌ Fail
                │               │             │           │           │
  Improved CV   │  ✅ Tốt       │  ✅ Tốt     │  ⚠️ OK    │  ⚠️ Kém  │  ❌ Fail
  (Phase 1+2)   │               │             │           │           │
                │               │             │           │           │
  Deep Learning │  ✅ Tốt       │  ✅ Tốt     │  ✅ Tốt   │  ✅ Tốt  │  ⚠️ OK
  (Phase 3)     │               │             │           │           │
```

**Kết luận:** Giải pháp hiện tại chỉ hoạt động tốt với trường hợp đơn giản nhất (ảnh phẳng, nền sạch, đủ 4 góc). Với dataset thực tế đa dạng như của bạn, cần ít nhất Phase 1 + Phase 2 để đạt tỷ lệ thành công ~60-70%, và Phase 3 (deep learning) để đạt >85%.
