import cv2
import sys
import os

def run_uvdoc_only():
    print("====================================")
    print(" Ví dụ: Chạy DUY NHẤT Neural UVDoc (Bỏ qua Detection & Enhancement) ")
    print("====================================")
    
    # Thư mục gốc dự án thiết lập đường dẫn cho import
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.join(base_dir, "Pipeline With ML")
    if pipeline_dir not in sys.path:
        sys.path.append(pipeline_dir)
    
    # Import UVDocDewarper trực tiếp từ bộ mã dự án
    try:
        from step2_uvdoc_dewarper import UVDocDewarper
    except ImportError as e:
        print(f"❌ Lỗi import UVDocDewarper: {e}")
        return

    # Đường dẫn ảnh gốc (Cong)
    image_path = os.path.join(base_dir, "image", "curved", "0000.jpg")
    print(f"[1] Đọc ảnh đầu vào: image/curved/0000.jpg")
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Không thể đọc ảnh gốc! Vui lòng kiểm tra lại đường dẫn.")
        return
        
    print(f"    Kích thước ảnh ban đầu: {img.shape[1]}x{img.shape[0]}")
    
    # Khởi tạo thuật toán UVDoc (Sẽ tự load file model best_model.pkl)
    print("\n[2] Khởi tạo mô hình nắn giấy 3D bằng mạng Neural (UVDoc)...")
    uvdoc = UVDocDewarper()
    
    # Đưa thẳng ảnh vào hàm dewarp mà không dùng YOLO cắt góc hay tiền xử lý cường độ sáng nào
    print("\n[3] Bắt đầu đẩy ảnh qua Neural Network để nắn phẳng (Suy luận Lưới 2D)...")
    result_img = uvdoc.dewarp(img)
    
    # Lưu kết quả
    save_dir = os.path.join(pipeline_dir, "result")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "curved_0000_uvdoc_only.jpg")
    cv2.imwrite(save_path, result_img)
    
    print("\n✅ Hoàn tất nắn cong!")
    print(f"Tất cả quy trình (cắt góc, làm nét) đã bị lược bỏ. Đầu ra thô của mạng UVDoc đã được lưu vào:")
    print(f"👉 {save_path}")

if __name__ == "__main__":
    run_uvdoc_only()
