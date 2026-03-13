import cv2
import os
import numpy as np
from scanner import DocumentScanner

def main():
    # Khởi tạo đối tượng nhận diện
    doc_scanner = DocumentScanner()
    
    # Ở đây thay bằng đường dẫn ảnh thực tế của bạn
    sample_image = "sample_receipt.jpg" 
    
    if not os.path.exists(sample_image):
        print(f"Vui lòng đặt một ảnh mẫu tên là '{sample_image}' vào thư mục này để chạy thử nghiệm.")
        # Tạo sẵn 1 ảnh mẫu xám trống để tránh lỗi crash nếu chưa có ảnh
        cv2.imwrite(sample_image, np.ones((800, 600, 3) if 'np' in globals() else (800, 600, 3), dtype=np.uint8) * 255)
        print(f"Đã tạo file ảnh nháp {sample_image}, vui lòng trỏ tới ảnh thật của bạn.")
        return

    print("Đang phân tích...")
    result_img, edged_img, corners = doc_scanner.scan(sample_image)

    if result_img is not None:
        print("=> Phân tích xong: ĐÃ TÌM THẤY BIÊN 4 GÓC!")
        # Hiển thị ảnh (thu nhỏ lại để vừa màn hình)
        result_show = cv2.resize(result_img, (600, int(600 * result_img.shape[0] / result_img.shape[1])))
        cv2.imshow("Result (Press any key to close)", result_show)
        cv2.imshow("Canny Edges", edged_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("=> Phân tích xong: KHÔNG TÌM THẤY 4 GÓC. Hãy thay đổi Background ảnh hoặc tinh chỉnh tham số Canny.")

if __name__ == "__main__":
    main()
