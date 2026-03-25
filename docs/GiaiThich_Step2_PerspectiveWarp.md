# Giải thích và Phân tích Lỗi Step 2 (Perspective Warp) sau khi chạy U²-Net

Gần đây bạn nhận thấy hiện tượng: **"Sau bước U²-Net thì Step 2 (Perspective Warp) đang gặp vấn đề / báo lỗi"**. Tài liệu này sẽ giải thích cặn kẽ nguyên lý học thuật đằng sau hiện tượng này, đồng thời mô hình hóa cách Step 2 hoạt động để bạn hiểu rõ bản chất của Pipeline.

---

## 1. TẠI SAO STEP 2 LẠI "LỖI / BỎ QUA" SAU KHI DÙNG U²-NET?

Trong `main.py`, khi bạn kích hoạt luồng U²-Net (`--u2net`), có một đoạn code chủ đích vô hiệu hóa Step 2:
```python
if result.get('u2net_doc') is not None:
    img_for_dewarp = result['u2net_doc']
    corners_for_dewarp = None # Tố tình Tắt Perspective
```
Khi `corners = None` được truyền vào `PerspectiveTransformer`, hàm sẽ báo: *"❌ [Step 2] Không đủ 4 điểm góc. Bỏ qua Perspective Transform."* 

**Đây không phải là lỗi rác (Bug), mà là Thiết kế An toàn (Fail-safe Design) với lý do sau:**
- **Bản chất của U²-Net:** U²-Net không trả về 4 tọa độ góc vuông. Nó trả về một "cái bóng" (Mask) cắt ôm sát đường biên cong lượn của tờ giấy nhàu nát/gập gáy. 
- **Sự xung đột Toán học:** Perspective Warp (Bước 2) **bắt buộc** phải nhận vào 4 điểm thẳng tắp để chiếu lên 1 hình chữ nhật. Nếu chúng ta cố ép cái mask lượn sóng của U²-Net thành 4 điểm và dùng Perspective Warp, kết quả tài liệu sẽ bị **kéo giãn bẹp dúm bóp méo** (giống như bạn cầm 4 góc của cái áo nhăn nhúm và kéo dãn nó ra một cách thô bạo).
- **Giải pháp CŨ:** Đối với đầu ra của U²-Net, ta phải bypass đoạn Linear Perspective (2a) và đẩy nó thẳng vào mạng AI nặn vật lý 3D **Text-line Dewarping (2b)** hoặc **UVDoc (2c)** để ủi phẳng mượt mà.

---

## 1.5 CẬP NHẬT MỚI NHẤT (SMART BYPASS & ANTI-PINCH)

Trong các file cập nhật gần nhất, thuật toán đã thông minh hơn rất nhiều. Việc loại bỏ Step 2 chỉ còn xảy ra nếu **thực sự quyển sách bị cong**.

Nếu U²-Net cắt ra một thẻ Căn Cước hoặc 1 phong bì phẳng góc nghiêng, góc uốn (Curve) bằng 0. Nếu khi ấy ta lôi UVDoc ra dùng, lưới nơ-ron của nó sẽ bị **Ảo giác Cánh Cung (Pinch Effect)** và thắt eo mảnh giấy lại!

Vì vậy, hệ thống đã trang bị Cơ chế chống Véo (Anti-Pinch):
- **Toán Học Không Gian (Intersection over Union - IoU):**
  - Hệ thống nhặt 4 góc ảo của u2net và kéo căng 4 đường thẳng tạo thành 1 khung Đa Giác lý tưởng (Ideal Polygon).
  - So sánh Diện Tích của Đa Giác này so với Mask cực nhỏ do U2Net tỉa.
  - Nếu `IoU > 94%`: Tài liệu lấp đầy 94% khung đa giác -> Thẳng cạnh! -> **HỦY BỎ NEURAL UVDOC**, kích hoạt lại Phương Trình Perspective Chiếu Hình Học (Step 2a) để tạo mảnh cắt sắc nét.
  - Nếu `IoU < 94%`: Tài liệu hụt viền (Cong võng xuống) -> Bỏ Step 2a, tiếp tục cho UVDoc bẻ cong tọa độ 3D.

---

## 2. MÔ HÌNH HÓA TOÁN HỌC: BƯỚC 2 (PERSPECTIVE WARP) HOẠT ĐỘNG NHƯ THẾ NÀO?

Để giúp bạn đưa vào báo cáo làm dẫn chứng, dưới đây là mô hình cơ chế hoạt động của Bước 2 khi nó hoạt động bình thường (với sự hỗ trợ của dò 4 góc từ DocAligner hoặc Canny fallback).

### A. Phương trình chiếu (Projective Transformation Matrix)
Mục tiêu là ánh xạ một tứ giác bất kỳ trên ảnh camera $P = \{P_1, P_2, P_3, P_4\}$ thành một hình chữ nhật phẳng $P' = \{P'_1, P'_2, P'_3, P'_4\}$ nhìn từ trên xuống (Bird's-eye view). 

Sự biến đổi của mỗi điểm pixel $(x, y)$ thành tọa độ mới $(x', y')$ được tính bằng nhân ma trận $3 \times 3$:

$$
\begin{bmatrix} x' w \\ y' w \\ w \end{bmatrix} = \begin{bmatrix} m_{11} & m_{12} & m_{13} \\ m_{21} & m_{22} & m_{23} \\ m_{31} & m_{32} & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

Suy ra tọa độ cuối cùng trên ảnh 2D là:
$$ x' = \frac{m_{11}x + m_{12}y + m_{13}}{m_{31}x + m_{32}y + 1} $$
$$ y' = \frac{m_{21}x + m_{22}y + m_{23}}{m_{31}x + m_{32}y + 1} $$

### B. Minh họa Luồng Dữ Liệu (Algorithm Flow)

```mermaid
graph TD
    A[Ảnh gốc chụp chéo] -->|Step 1 DocAligner| B(Sắp xếp 4 điểm: TL, TR, BR, BL)
    B --> C{Tính kích thước Max Height / Max Width}
    
    C -->|Width| W[W = max(TopEdge, BottomEdge)]
    C -->|Height| H[H = max(LeftEdge, RightEdge)]
    
    W --> E[Lập khung ảnh đích chuẩn: <br/> 0,0 đến W,H]
    H --> E
    
    B --> M
    E --> M((Giải ma trận cv2.getPerspectiveTransform))
    
    M --> F[Áp dụng cv2.warpPerspective]
    F --> G[Ảnh chữ nhật Duỗi phẳng]
```

### C. Cơ chế giới hạn (Tại sao phương trình này chết trước giấy nhàu/giấy cong?)
Ma trận $M$ giữ nguyên **tính thẳng** của các đoạn thẳng (Segment Linearity). Nghĩa là: Cạnh của tờ giấy dù chéo, sau khi Warp vẫn sẽ tạo thành nét thẳng góc vuông.
$\rightarrow$ **Do góc gáy sách là một ĐƯỜNG CONG**, Perspective Warp sẽ bẻ cong ngược lại hình ảnh chữ bên trong tờ giấy khiến chúng méo xệch. Đó là lý do U²-Net phải đi kèm với mạng lưới biến dạng lưới **Dewarp Grid (Layer 2c)** thay vì hình học thuần túy.
