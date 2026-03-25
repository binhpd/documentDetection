import cv2
import numpy as np
import os

def auto_crop_and_pad(image_path, output_path, padding_size=30):
    print(f"Bắt đầu xử lý ảnh: {image_path}")
    # Đọc ảnh gốc
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh!")
        return

    # Chuyển xám để xử lý
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Binarize (Nhị phân hóa để lấy chữ)
    # Dùng ngưỡng động kết hợp Otsu hoặc Adaptive
    # Ở đây do trên web hoặc ảnh scan, chữ có thể là màu tối nền sáng. 
    # Ta dùng cv2.THRESH_BINARY_INV + OTSU để chữ thành trắng, nền đen
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Áp dụng Morphological Operations (phóng to khối chữ dính liền vào nhau)
    # Dùng kernel nằm ngang dài và dọc để gộp chữ thành đoạn/khối (Text Block)
    kernel_size = (50, 50) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Nối các chữ lại với nhau
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # 3. Tìm các contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Không tìm thấy text block nào.")
        return
    
    # 4. Tìm Bounding Box bao trọn tất cả các phần chữ (lớn hơn 1 diện tích nhất định)
    # Hoặc đơn giản là lấy tọa độ min/max của tất cả bounding box
    min_x, min_y = img.shape[1], img.shape[0]
    max_x, max_y = 0, 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Bỏ qua nhiễu nhỏ
        if w > 20 and h > 20: 
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            if x + w > max_x: max_x = x + w
            if y + h > max_y: max_y = y + h
            
    # Thêm một chút xíu padding mềm cho Box khỏi cắt sát chữ quá
    margin = 5
    min_x = max(0, min_x - margin)
    min_y = max(0, min_y - margin)
    max_x = min(img.shape[1], max_x + margin)
    max_y = min(img.shape[0], max_y + margin)
    
    # 5. Cắt khối văn bản (Crop)
    cropped_img = img[min_y:max_y, min_x:max_x]
    
    # Lấy màu nền (Xác định màu trung bình ở 4 góc của ảnh crop hoặc dùng trắng)
    # Thường với ảnh document ta nên pad màu Trắng hoặc màu xám trung bình của lề text
    bg_color = [255, 255, 255] # Nền trắng
    
    # 6. Thêm Padding mới đều đẹp, phẳng phiu cho xung quanh
    # cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
    final_img = cv2.copyMakeBorder(cropped_img, padding_size, padding_size, padding_size, padding_size, 
                                   cv2.BORDER_CONSTANT, value=bg_color)
    
    # Lưu file kết quả
    cv2.imwrite(output_path, final_img)
    print(f"Đã lưu kết quả tại: {output_path}")

if __name__ == "__main__":
    input_img = "/Users/binhpham/Documents/Study/MSE/Xử lý ảnh Video/Bài tập cuối kỳ/Nhóm 6/PipelineRunMobile/mobile_result/step2_Dewarp_uvdoc15MB.jpg"
    output_img = "/Users/binhpham/Documents/Study/MSE/Xử lý ảnh Video/Bài tập cuối kỳ/Nhóm 6/PipelineRunMobile/mobile_result/step2b_Aligned_Padding.jpg"
    
    auto_crop_and_pad(input_img, output_img, padding_size=40)
