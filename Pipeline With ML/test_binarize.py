import cv2
import numpy as np

def new_binarize(img_path):
    # Đọc ảnh noshadow
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None: return

    # 1. Gamma Correction (kéo gắt nét phai)
    gamma = 2.0
    gray_gamma = np.array(255 * (gray / 255) ** gamma, dtype=np.uint8)

    # 2. Gaussian Blur cực nhẹ để khử nhiễu răng cưa
    blurred = cv2.GaussianBlur(gray_gamma, (3, 3), 0)

    # 3. Thuật toán Otsu (Chia nền trắng tinh / Chữ đen kịt)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 4. Nối nét đứt, điền khuyết (Morphological Closing trên nền chữ đen)
    # Binary: Chữ là đen (0), nền là trắng (255)
    # Ta đảo ngược để chữ thành (255) dễ gọt
    inv = cv2.bitwise_not(binary)
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Nối nét đứt bên trong thân chữ
    inv_connected = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel_connect)
    # Bơm phồng chữ mờ lên thêm một chút
    inv_bold = cv2.dilate(inv_connected, kernel_connect, iterations=1)

    # Trả lại nền trắng
    final_img = cv2.bitwise_not(inv_bold)
    cv2.imwrite("test_final.jpg", final_img)

new_binarize("result/fold_0016_step3_3_noshadow.jpg")
