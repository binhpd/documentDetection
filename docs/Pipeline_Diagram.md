# Kiến Trúc Luồng Thuật Toán (Mới Nhất)

Sơ đồ Mermaid dưới đây minh họa toàn bộ vòng đời phân bổ dữ liệu và logic xuyên suốt của **Hybrid ML Document Scanner Pipeline**.

```mermaid
graph TD
    %% Định nghĩa hình dáng
    classDef step fill:#d1e7dd,stroke:#0f5132,stroke-width:2px;
    classDef input fill:#f8d7da,stroke:#842029,stroke-width:2px;
    classDef process fill:#fff3cd,stroke:#664d03,stroke-width:2px;
    classDef output fill:#cff4fc,stroke:#055160,stroke-width:2px;
    classDef ai fill:#e2e3e5,stroke:#41464b,stroke-width:2px,stroke-dasharray: 5 5;

    %% Bắt đầu
    IN[Ảnh Gốc Chụp Từ Mobile]:::input --> S1[STEP 1: Document Detection]:::step

    %% Step 1 Mạch
    subgraph S1_Block [Bước 1: Machine Learning Segmentation]
        S1 --> S1_A{Chọn Luồng AI}:::process
        
        S1_A -->|--u2net| U2N[Luồng A: U²-Net Rembg]:::ai
        S1_A -->|mặc định| DOC[Luồng B: DocAligner SOTA]:::ai
        
        U2N --> MS1(Tạo Mask Dựa Trên Ngữ Nghĩa)
        DOC --> MS2(Tạo Bounding Box minAreaRect)
        
        MS1 --> CO1[Tọa độ 4 Góc + Mask Phân vùng]:::process
        MS2 --> CO1
    end

    %% Rẽ qua Step 2
    CO1 --> S2[STEP 2: Geometric Dewarping]:::step

    %% Step 2 Mạch
    subgraph S2_Block [Bước 2: Phục Hồi Hình Học Khôn Ngoan]
        S2 --> IOU{Chốt Chặn Kiểm Tra:<br/>Tính Diện Tích IoU Mask <br/>vs Đa Giác 4 Góc lý tưởng}:::process
        
        IOU -- IoU > 94% <br/>(Tài liệu Phẳng) --> PER[Perspective Transform Matrix]:::process
        IOU -- IoU < 94% <br/>(Giấy Cong/Nhăn) --> UVD[Fallback tới UVDoc Neural Grid]:::ai
        
        PER --> FL1(Ảnh Chữ Nhật Quét Phẳng Tắp)
        UVD --> FL2(Ảnh Uốn Ngược Xóa Rãnh Gáy Sách)

        FL1 --> DEW[Ảnh Đã Trải Rộng Khung A4]:::process
        FL2 --> DEW
    end

    %% Rẽ qua Step 3
    DEW --> S3[STEP 3: Image Enhancement Endpoint]:::step

    %% Step 3 Mạch
    subgraph S3_Block [Bước 3: Tăng Cường Quang Học Micro Pixel]
        S3 --> GLA[Khử Lóa Flash: <br/>cv2.inpaint Telea]:::process
        GLA --> BLU[Sắc Lẹm Chữ Rung Tay: <br/>Unsharp Masking]:::process
        
        BLU --> RGB[Tách Nguồn Sáng Kép RGB: <br/>Division Shadow Normalization]:::process
        
        RGB --> FIN[Binarization Cuối Cùng:<br/>Piecewise Linear Contrast Stretch]:::process
    end

    %% Kết thúc
    FIN --> OUT[Bản Scan Enterprise Hoàn Mỹ <br/> (Sạch bóng, Viền nguyên vẹn, Siêu Nhẹ)]:::output
```

**Cách Đọc Biểu Đồ:**
- Khối nét đứt nền xám (`Luồng A`, `Luồng B`, `Fallback UVDoc`) đại diện cho các Điểm Tính Toán Nơ-ron bằng AI (Machine Learning / Deep Learning).
- Khối nền vàng đại diện cho các bước tính toán Thuật Toán tĩnh (Toán C++ OpenCV).
- Tính năng **Chốt Chặn Kiểm tra IoU** là nấc thang bảo chứng rằng máy tính không bao giờ lạm dụng cấu trúc 3D rườm rà lên một tờ biên lai thẳng thớm.
