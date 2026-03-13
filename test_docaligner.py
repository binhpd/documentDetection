import cv2
import numpy as np

try:
    from docaligner import DocAligner
    
    # Initialize model
    model = DocAligner()
    
    # Read image
    img_path = "/Users/binhpham/Documents/Study/MSE/Xử lý ảnh Video/Bài tập cuối kỳ/Nhóm 6/image/perspective/0009.jpg"
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Failed to read image at: {img_path}")
    else:
        print("Running DocAligner...")
        # Get polygon (4 corners)
        polygon = model(img)
        print("Model output:", polygon)
        print("Output shape/type:", type(polygon))
        
except Exception as e:
    print(f"Error testing DocAligner: {e}")
