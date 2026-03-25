import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Cấu hình đường dẫn ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pipeline_dir = os.path.join(base_dir, "Pipeline With ML")
uvdoc_repo = os.path.join(pipeline_dir, "UVDoc_repo")
sys.path.append(pipeline_dir)
sys.path.append(uvdoc_repo)

from utils import load_model, IMG_SIZE

def plot_3d_mesh():
    result_dir = os.path.join(base_dir, "PipelineRunMobile", "mobile_result")
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. Khởi tạo thiết bị và tải mô hình
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    fp16_model_path = os.path.join(uvdoc_repo, "model", "best_model_fp16.pkl")
    orig_model_path = os.path.join(uvdoc_repo, "model", "best_model.pkl")
    
    print("Đang tải UVDoc model...")
    model = load_model(orig_model_path)
    model.to(device).half()
    model.load_state_dict(torch.load(fp16_model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 2. Đọc và chuẩn bị ảnh
    test_img_path = os.path.join(base_dir, "image", "fold", "0001.jpg")
    img = cv2.imread(test_img_path)
    if img is None:
        print("Lỗi: Không thể tải ảnh", test_img_path)
        return
        
    print(f"Kích thước ảnh gốc: {img.shape}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp_resized = cv2.resize(img_rgb, tuple(IMG_SIZE))
    inp_tensor = torch.from_numpy(inp_resized.transpose(2, 0, 1)).unsqueeze(0).to(device).half()
    
    # 3. Suy luận Mạng Neural để lấy Lưới 3D
    print("Đang suy luận mạng UVDoc...")
    with torch.no_grad():
        _, point_positions3D = model(inp_tensor)
        
    # point_positions3D có kích thước: (Batch, 3, Grid_H, Grid_W)
    point3d = point_positions3D[0].detach().cpu().float().numpy() # shape: (3, H, W)
    
    # Các kênh là: (X, Y, Z) hoặc tương tự. Thường kênh 0 là X, 1 là Y, 2 là Z (chiều sâu).
    print(f"Kích thước lưới 3D: {point3d.shape}")
    
    X = point3d[0, :, :]
    Y = point3d[1, :, :]
    Z = point3d[2, :, :]
    
    # 4. Vẽ biểu đồ 3D Mesh
    print("Đang tạo biểu đồ 3D mặt giấy...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot bề mặt giấy
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    # Có thể vẽ dạng lưới xương (wireframe) để nhìn rõ cách mạng bẻ cong
    ax.plot_wireframe(X, Y, Z, color='black', linewidth=0.3, alpha=0.5)
    
    ax.set_title('UVDoc 3D Paper Neural Mesh Prediction', fontsize=16)
    ax.set_xlabel('Camera X')
    ax.set_ylabel('Camera Y')
    ax.set_zlabel('Depth (Z)')
    
    # Góc nhìn mặc định cho biểu đồ 3D
    ax.view_init(elev=50, azim=45)
    
    # Lưu kết quả
    output_path = os.path.join(result_dir, "step2_3D_Mesh_Visualized.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Đã vẽ và lưu ảnh 3D Mesh thành công tại: {output_path}")

if __name__ == "__main__":
    plot_3d_mesh()
