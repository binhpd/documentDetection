import os
import sys
import torch

def quantize_uvdoc_to_fp16():
    # Thư mục chứa model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "Pipeline With ML", "UVDoc_repo", "model")
    orig_model_path = os.path.join(model_dir, "best_model.pkl")
    fp16_model_path = os.path.join(model_dir, "best_model_fp16.pkl")
    
    if not os.path.exists(orig_model_path):
        print(f"❌ Không tìm thấy model gốc tại: {orig_model_path}")
        return
        
    orig_size_mb = os.path.getsize(orig_model_path) / (1024 * 1024)
    print(f"[!] Kích thước model gốc (Float32): {orig_size_mb:.2f} MB")
    
    print("[...] Đang tải model gốc vào RAM...")
    
    # Import UVDoc utils để dùng hàm load_model
    sys.path.append(os.path.join(base_dir, "Pipeline With ML", "UVDoc_repo"))
    from utils import load_model
    
    try:
        model = load_model(orig_model_path)
    except Exception as e:
        print(f"Lỗi load model: {e}")
        return
        
    print("[...] Đang thực hiện Quantization trọng số sang Float16 (FP16)...")
    model.half() # Convert model weights to FP16
    
    print(f"[...] Đang lưu model tĩnh FP16 ra ổ cứng: {fp16_model_path}")
    torch.save(model.state_dict(), fp16_model_path)
    
    fp16_size_mb = os.path.getsize(fp16_model_path) / (1024 * 1024)
    print(f"✅ Kích thước model sau khi nén (Float16): {fp16_size_mb:.2f} MB")
    
    reduction = 100 - (fp16_size_mb / orig_size_mb * 100)
    print(f"📉 Đã giảm được {reduction:.2f}% dung lượng!")

if __name__ == "__main__":
    quantize_uvdoc_to_fp16()
