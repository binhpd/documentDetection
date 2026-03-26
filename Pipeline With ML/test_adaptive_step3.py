import cv2
import os
from step3_enhancer import DocumentEnhancer

def test():
    enhancer = DocumentEnhancer()
    test_files = [
        "result/fold_0001_step2_dewarped.jpg",
        "result/testing_IMG_5420_step2_dewarped.jpg"
    ]
    
    for f in test_files:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            continue
            
        print(f"\n[Test] Processing {f}")
        img = cv2.imread(f)
        if img is None:
            print(f"Failed to read {f}")
            continue
            
        base_name = os.path.basename(f).replace("_step2_dewarped.jpg", "")
        save_prefix = f"result/{base_name}_test"
        
        # Gọi thuật toán chính
        final_img = enhancer.enhance(img, save_prefix=save_prefix, mode="color")
        print(f"-> Successfully processed {base_name}")

if __name__ == "__main__":
    test()
