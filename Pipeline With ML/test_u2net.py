import sys
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
from page_dewarp.image import WarpedImage
from page_dewarp.options import Config

from main import get_images_in_category

def main():
    parser = argparse.ArgumentParser(description="Test U2-Net Background Removal & Dewarping")
    parser.add_argument("folder", help="Tên thư mục ảnh (VD: curved, perspective)")
    parser.add_argument("index", type=int, default=0, nargs="?", help="Số thứ tự ảnh")
    args = parser.parse_args()

    images = get_images_in_category(args.folder)
    if not images:
        print(f"❌ Không tìm thấy thư mục hoặc ảnh trong: {args.folder}")
        return
        
    idx = min(args.index, len(images) - 1)
    image_path = images[idx]
    
    print(f"📂 Đã chọn ảnh: {image_path}")
    print(f"🚀 Bắt đầu đọc ảnh...")
    img = cv2.imread(image_path)
    if img is None:
        return print("❌ Không tìm thấy hoặc bị lỗi khởi tạo ảnh!")

    print("🧠 BƯỚC 1: DÙNG U²-NET ĐỤC NỀN & TRÍCH XUẤT VIỀN CONG TÀI LIỆU...")
    # remove() của rembg nhận vào byte hoặc mảng numpy RGB, trả ra ảnh PNG có kênh Alpha
    # Nó sẽ tự động phân tích và loại bỏ hoàn toàn sạch bóng sàn nhà/mặt bàn, trả lại 1 cuốn sách cong nguyên vẹn
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    subject = remove(rgb_img) 
    
    # Ép kênh Alpha (trong suốt) thành nền Trắng tinh khôi để bảo vệ nét chữ
    alpha_channel = subject[:, :, 3]
    rgb_output = subject[:, :, :3]
    white_background = np.ones_like(rgb_output, dtype=np.uint8) * 255
    foreground_mask = alpha_channel[:, :, np.newaxis] / 255.0
    
    # Hòa trộn: Lấy sách ghép lên nền trắng
    pure_document = (rgb_output * foreground_mask + white_background * (1 - foreground_mask)).astype(np.uint8)
    pure_document_bgr = cv2.cvtColor(pure_document, cv2.COLOR_RGB2BGR)

    cv2.imwrite("01_u2net_extracted_curved_doc.jpg", pure_document_bgr)
    print("✅ Xong! Đã bóc sạch nền và giữ nguyên nét gợn sóng (lưu tại 01_u2net_extracted_curved_doc.jpg)")

    print("\n🧠 BƯỚC 2: NÉM ẢNH ĐÃ BÓC NỀN VÀO PAGE-DEWARP ĐỂ LÀ PHẲNG...")
    # Vì nền đã trắng tinh và chả còn rác nhiễu, page-dewarp sẽ bắt lưới (mesh) gợn sóng cực trơn tru
    config = Config()
    
    # Ép thư viện nhả ra quá trình "Tưởng tượng đường cong"
    if hasattr(config, 'debug_level'):
        config.debug_level = 2
    if hasattr(config, 'DEBUG_LEVEL'):
        config.DEBUG_LEVEL = 2

    plots = [("Ảnh Gốc", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), 
             ("U²-Net Lọc Nền", cv2.cvtColor(pure_document_bgr, cv2.COLOR_BGR2RGB))]
             
    try:
        warped = WarpedImage("01_u2net_extracted_curved_doc.jpg", config=config)
        print(f"🎉 Chạy thuật toán page-dewarp xong!")
        
        # Load các ảnh Debug vẽ đường Line của thuật toán (nếu sinh ra thành công)
        before_file = "01_u2net_extracted_curved_doc_debug_4_keypoints_before.png"
        after_file = "01_u2net_extracted_curved_doc_debug_5_keypoints_after.png"
        
        if os.path.exists(before_file):
            plots.append(("Lưới ảo 3D ban đầu", cv2.cvtColor(cv2.imread(before_file), cv2.COLOR_BGR2RGB)))
            
        if os.path.exists(after_file):
            plots.append(("Lưới ảo sau khi Là", cv2.cvtColor(cv2.imread(after_file), cv2.COLOR_BGR2RGB)))

        if hasattr(warped, 'outfile') and warped.outfile and os.path.exists(str(warped.outfile)):
            dewarp_res = cv2.imread(str(warped.outfile))
            if dewarp_res is not None:
                plots.append(("Page-Dewarp Là Phẳng", cv2.cvtColor(dewarp_res, cv2.COLOR_BGR2RGB)))
    except Exception as e:
        print(f"Lỗi Dewarping: {e}")

    print("\n🖼️ ĐANG TẠO BẢNG ĐÁNH GIÁ CHẤT LƯỢNG VÀ HIỂN THỊ...")
    n = len(plots)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1: axes = [axes]
    
    for ax, (title, plot_img) in zip(axes, plots):
        ax.imshow(plot_img)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")
        
    plt.tight_layout()
    # Tự động xuất file độ nét cao nộp cho giáo viên / báo cáo
    report_file = "Report_U2Net_Dewarp_Demo.png"
    fig.savefig(report_file, dpi=300, bbox_inches='tight')
    print(f"🎁 Đã đóng gói giao diện so sánh toàn cảnh MỚI NHẤT thành file: {report_file}")
    plt.show()

if __name__ == "__main__":
    main()
