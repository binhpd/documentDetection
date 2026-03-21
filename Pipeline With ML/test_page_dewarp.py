import sys
import cv2
import os

from main import DocumentDetector

def main():
    image_path = "../image/perspective/0012.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Test image not found at {image_path}")
        return

    print(f"🚀 Bắt đầu tích hợp page_dewarp vào Pipeline ML với ảnh: {image_path}")
    
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Lỗi đọc ảnh!")
        return

    # 1. Khởi tạo Pipeline (Bật cờ enable_ml và use_ml_dewarp để kích hoạt page-dewarp)
    print("\n--- [STEP 1] CẮT TÀI LIỆU RỜI KHỎI NỀN ---")
    detector = DocumentDetector(enable_ml=True, use_ml_dewarp=True)
    
    # Gọi hàm detect để tự động tìm 4 góc (Bao gồm Resize -> Gray -> Blur -> Canny -> DL Segmentation)
    result = detector.detect(img)
    corners = result.get('corners')
    
    if corners is None:
        print("❌ Không tìm được 4 góc tài liệu. Dừng pipeline.")
        return
        
    print(f"✅ Đã tìm được 4 góc tọa độ bằng phương pháp: {result.get('method')}")
    
    # 2. Chạy Step 2: Uốn phẳng
    # (Hàm dewarp() của MLDewarper đã được tích hợp sẵn 2 việc: Crop ảnh theo 4 góc -> Giao cho page-dewarp phân tích độ cong)
    print("\n--- [STEP 2] UỐN PHẲNG BẰNG TEXT-LINE (PAGE-DEWARP) ---")
    try:
        if hasattr(detector.transformer, 'dewarp'):
            warped = detector.transformer.dewarp(img, corners)
            
            print("✅ Xử lý uốn phẳng thành công!")
            
            # Lưu và kiểm tra kết quả
            out_path = "0012_dewarped_result.jpg"
            cv2.imwrite(out_path, warped)
            print(f"🎉 Đã lưu kết quả hoàn chỉnh tại: {out_path}")
        else:
            print("❌ Transformer không có hàm dewarp, vui lòng bật cờ use_ml_dewarp=True.")
            
    except Exception as e:
        print(f" Lỗi trong quá trình dewarping: {e}")

if __name__ == "__main__":
    main()
