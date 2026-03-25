import base64
import zlib
import urllib.request
import urllib.error
import sys

mermaid_code = """graph TD
    classDef step fill:#d1e7dd,stroke:#0f5132,stroke-width:2px;
    classDef input fill:#f8d7da,stroke:#842029,stroke-width:2px;
    classDef process fill:#fff3cd,stroke:#664d03,stroke-width:2px;
    classDef output fill:#cff4fc,stroke:#055160,stroke-width:2px;
    classDef ai fill:#e2e3e5,stroke:#41464b,stroke-width:2px,stroke-dasharray: 5 5;

    IN[Ảnh Gốc Chụp Từ Mobile]:::input --> S1[STEP 1: Document Detection]:::step

    subgraph S1_Block [Bước 1: Machine Learning Segmentation]
        S1 --> S1_A{Chọn Luồng AI}:::process
        
        S1_A -->|--u2net| U2N[Luồng A: U²-Net Rembg]:::ai
        S1_A -->|mặc định| DOC[Luồng B: DocAligner SOTA]:::ai
        
        U2N --> MS1(Tạo Mask Dựa Trên Ngữ Nghĩa)
        DOC --> MS2(Tạo Bounding Box minAreaRect)
        
        MS1 --> CO1[Tọa độ 4 Góc + Mask Phân vùng]:::process
        MS2 --> CO1
    end

    CO1 --> S2[STEP 2: Geometric Dewarping]:::step

    subgraph S2_Block [Bước 2: Phục Hồi Hình Học Khôn Ngoan]
        S2 --> IOU{Chốt Chặn Kiểm Tra:<br/>Tính Diện Tích IoU Mask <br/>vs Đa Giác 4 Góc lý tưởng}:::process
        
        IOU -- IoU > 94% <br/>(Tài liệu Phẳng) --> PER[Perspective Transform Matrix]:::process
        IOU -- IoU < 94% <br/>(Giấy Cong/Nhăn) --> UVD[Fallback tới UVDoc Neural Grid]:::ai
        
        PER --> FL1(Ảnh Chữ Nhật Quét Phẳng Tắp)
        UVD --> FL2(Ảnh Uốn Ngược Xóa Rãnh Gáy Sách)

        FL1 --> DEW[Ảnh Đã Trải Rộng Khung A4]:::process
        FL2 --> DEW
    end

    DEW --> S3[STEP 3: Image Enhancement Endpoint]:::step

    subgraph S3_Block [Bước 3: Tăng Cường Quang Học Micro Pixel]
        S3 --> GLA[Khử Lóa Flash: <br/>cv2.inpaint Telea]:::process
        GLA --> BLU[Sắc Lẹm Chữ Rung Tay: <br/>Unsharp Masking]:::process
        
        BLU --> RGB[Tách Nguồn Sáng Kép RGB: <br/>Division Shadow Normalization]:::process
        
        RGB --> FIN[Binarization Cuối Cùng:<br/>Piecewise Linear Contrast Stretch]:::process
    end

    FIN --> OUT[Bản Scan Enterprise Hoàn Mỹ <br/> Sạch bóng, Viền nguyên vẹn, Siêu Nhẹ]:::output"""

compressed = zlib.compress(mermaid_code.encode('utf-8'), 9)
b64 = base64.urlsafe_b64encode(compressed).decode('utf-8')
url = f"https://kroki.io/mermaid/png/{b64}"

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response, open('docs/Pipeline_Diagram_HighRes.png', 'wb') as out_file:
        out_file.write(response.read())
    print('Successfully downloaded docs/Pipeline_Diagram_HighRes.png')
except urllib.error.HTTPError as e:
    print(f'HTTP Error: {e.code}')
    print(e.read().decode('utf-8'))
    sys.exit(1)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
