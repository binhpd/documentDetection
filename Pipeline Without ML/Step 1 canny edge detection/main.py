import cv2
import os
import numpy as np
from scanner import DocumentScanner

import sys

def main():
    # Khởi tạo đối tượng nhận diện
    doc_scanner = DocumentScanner()
    
    if len(sys.argv) >= 3:
        category = sys.argv[1]
        img_idx = sys.argv[2]
        sample_image = os.path.join(os.path.dirname(__file__), "..", "..", "image", category, f"{int(img_idx):04d}.jpg")
    elif len(sys.argv) == 2:
        sample_image = sys.argv[1]
    else:
        sample_image = "sample_receipt.jpg" 
        
    print(f"File ảnh đầu vào: {sample_image}")
    
    if not os.path.exists(sample_image):
        print(f"Không tìm thấy file ảnh: '{sample_image}'")
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
