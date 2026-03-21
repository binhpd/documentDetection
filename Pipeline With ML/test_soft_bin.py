import cv2
import numpy as np

def new_binarize(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None: return

    smooth = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # Thử threshold mềm P_LOW, P_HIGH
    p_low = 140
    p_high = 220
    
    float_img = smooth.astype(float)
    stretched = (float_img - p_low) * (255.0 / (p_high - p_low))
    final_img = np.clip(stretched, 0, 255).astype(np.uint8)

    gaussian = cv2.GaussianBlur(final_img, (0, 0), 1.0)
    final_img = cv2.addWeighted(final_img, 1.5, gaussian, -0.5, 0)

    cv2.imwrite("test_final2.jpg", final_img)

new_binarize("result/fold_0016_step3_3_noshadow.jpg")
