# Giải Thích Chi Tiết: STEP 3 - TĂNG CƯỜNG CHẤT LƯỢNG ẢNH (Image Enhancement)

Ở bước này, bức ảnh tờ giấy tuy đã phẳng hình học (dewarped) nhưng bề mặt giấy vẫn loang lổ bóng râm (cầm điện thoại bị sấp bóng tay), có mảng sáng lóa do flash, và chữ viết tay thường bị mờ. Ta sẽ dùng Computer Vision để nâng tầm nó lên chuẩn Scanner công nghiệp (mực đen sắc lẹm, giấy trắng tinh không dính sạn đen).

---

## 1. Thuật toán Khôi Phục Lóa Sáng (Glare Inpainting)
Nếu chụp cận cảnh dưới ánh đèn LED, một mảng giấy bóng cứng sẽ sáng lóa che mất chữ. 

* **Toán học cốt lõi:** Phương trình truyền nhiệt hoặc Động lực học chất lưu Navier-Stokes.
* **Cơ chế thực thi:**
  1. Dùng `cv2.threshold` cắt lớp ở cường độ ánh sáng cực đỉnh ($> 250$). Chỉ những lỗ hổng do ánh chớp chói mắt (Flash) tạo ra mới lọt vào. Đặt tên là `clipping_mask`.
  2. Đo diện tích hạt chói. Nếu diện tích đốm sáng quá liều (quá lớn $>$ 5% bức ảnh) $\rightarrow$ Không thể cứu vãn vì nội suy sẽ sinh rác ảo giác.
  3. Nếu mảng lóa nhỏ: Gọi thuật toán `cv2.inpaint` với method Telea (dựa trên thuật toán Fast Marching). Nó sẽ copy màu rìa xám xịt xung quanh cái đốm sáng, vuốt nối màu loang dần vào tâm điểm sáng lóa để bít cái hố lóa đó lại một cách tự nhiên.
* **Input:** Ảnh chói lóa.
* **Output:** Ảnh lấp lóa hoàn hảo, màu tệp với xung quanh.

---

## 2. Làm Nét Chống Rung (Unsharp Masking)
Nhòe ảnh (Motion Blur) do rung tay (Camera Shake) hoặc lấy nét lệch.

* **Cơ chế thực thi:**
  - Làm nhòe ảnh một cách có chủ đích: `blur = cv2.GaussianBlur(img, (0,0), 3)`
  - Thuật toán Mask Nét Cắt: Lấy ảnh Gốc Trừ đi ảnh Nhòe $\rightarrow$ Sự khác biệt dôi ra chính là các "**Cạnh Biên**" (Edges) mỏng manh của sợi chữ.
  - Phép tính Khuếch Đại (Kéo Khung `cv2.addWeighted`): Cộng dồn lượng viền chữ đó đè thẳng lên ảnh gốc. Viền chữ đen lập tức đâm sọc tương phản mạnh.
* **Output:** Chữ in đậm hơn, gắt cạnh hơn. Chữa cận thị ảnh hiệu quả.

---

## 3. Khử Bóng Râm Nửa Mùa Độc Lập Kênh (RGB Shadow Normalization)
Thuật toán chia ánh sáng đỉnh cao nhát phá vỡ 100% bóng đổ đen do thân mình/cánh tay che lên giấy khi chụp.

* **Sự Yếu kém của giải pháp Cũ:** Các thuật toán Otsu Local block thường dò bóng bằng việc chuyển cạn ảnh sang Trắng Đen (Gray), làm mất toàn bộ nhiệt độ màu, con mộc Đỏ dính bóng mờ dễ ngả sang màu Tím hoặc Xám xi măng.

* **Cơ chế Phương Pháp Kênh Độc Lập (Đỉnh cao):**
  1. Tách ảnh thành 3 dòng thực thể: Mảng Đỏ (R), Mảng Lục (G), Mảng Lam (B).
  2. Tạo Bản Đồ Nguồn Sáng Kép (Background Maps): Tại từng mảng, phóng to (Dilate) để ăn tươi rễ chữ mực mỏng đi. Sau đó phủ Gaussian Blur cỡ Đại ($51 \times 51$) dẹp hết các tàn dư cục bộ. Ta thu được 1 Bản đồ dốc đổ bóng MƯỢT MÀ, biểu đồ miêu tả sự lan tỏa của bóng tay trùm lên tờ giấy.
  3. Thay vì đem đi Trừ (Subtract), thuật toán dùng thuyết Chiếu Sáng Phản Tượng ($I = L \times R$). Ta thực hiện phép **CHIA**: Nghĩa là `255 - (255 - pixel_kênh_đó) * (255 / (background_map_kênh_đó))`.
  4. Trộn 3 dòng màu trở lại.
* **Kết quả Tuyệt Mật:** Toàn bộ bóng tay trên giấy lập tức bay màn vô hình. Tờ giấy xám ngoét biến thành giấy trắng sáng loáng (Bright White Paper). ĐẶC BIỆT NHẤT: Màu Xanh chữ viết tay, màu Đỏ con mộc son hoàn toàn Tươi Sống và tệp chính xác độ tương phản màu thực không suy suyển 1 hạt bit.

---

## 4. Phơi Sáng Thích Ứng \u0026 Cắt Cặn Tương Phản (Adaptive Soft Binarization \u0026 Denoising)
Nhị phân hóa (Binarize) để in ra giấy (Mực Đen, Không Nenn Xám) bằng Otsu thường gây mẻ rỗ chữ. Thuật toán cố định (Fixed Thresholds) lại thường xuyên lỗi nếu ảnh sáng tối không đều. Chúng tôi sử dụng **Tự Thích Ứng (Adaptive)** để giải quyết trọn vẹn.

* **Toán học cốt lõi:** Contrast Limited Adaptive Histogram Equalization (CLAHE) \u0026 Bách Phân Vị (Percentile Stretching).
* **Cơ chế 2 Chế Độ (B/W \u0026 Color):**
  - **B/W Mode:** 
    1. Chà mượt nền giấy bằng `Bilateral Filter` để loại nhiễu râm.
    2. Chạy `CLAHE` đẩy bật tương phản các chữ bút chì mờ nhạt chìm trong lều xám.
    3. Trích xuất **Adaptive Threshold** bằng việc đo Bách Phân Vị (2% điểm tối nhất thành lõi mực Đen $0$, 90% điểm sáng rực thành Nền Trắng $255$). Vùng giao thoa được giãn mượt.
    4. Hàn gắn các nứt gãy của nét chữ bằng `Morphological Close`.
  - **Color Mode (Bản màu):**
    1. Trích xuất không gian phân giải màu **LAB**. Tiến hành Blur cực gắt trên 2 kênh phân giải màu $A$ và $B$ (Đánh bay hoàn toàn rác hạt màu tím/xanh/đỏ trên bề mặt giấy trắng). Kênh độ sáng $L$ chứa viền chữ được bảo vệ tuyệt đối.
    2. Tự động nội suy kéo giãn dải Histogram để trắng nền phơi sáng điểm dư thừa.
    3. Đẩy nhẹ Saturation HSV thông minh (Chỉ đẩy ở các vùng có màu thực sự $>15$ để bảo tồn con mộc đỏ rực rõ mà không rớt màu vào nền giấy trắng).
* **Output:** Nền ngoài trắng phau như tuyết, chữ tuyệt đối không rỗ mẻ và không bị lẹm viền. Con dấu mộc và chữ ký màu hiển thị sắc nét đỉnh cao mà không còn bất kỳ tia nhiễu xám nào. Tự hào quét tài liệu đẳng cấp Enterprise.
