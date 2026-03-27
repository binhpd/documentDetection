import sys
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response

# Thêm đường dẫn tới thư mục Pipeline With ML
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(BASE_DIR, "..", "Pipeline With ML")
sys.path.append(PIPELINE_DIR)

from main import DocumentDetector
from corner_sorter import CornerSorter

app = FastAPI(title="DocScanner API")

# Khởi tạo detector một lần khi startup app để tiết kiệm tài nguyên
detector = DocumentDetector(enable_ml=True, use_docaligner=False, use_ml_dewarp=True, use_u2net=True, use_uvdoc=False)

@app.post("/api/scan")
async def scan_document(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")
    
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    orig = img.copy()
    
    # ── Chạy pipeline Step 1 ──
    # u2net được set = True mặc định
    result = detector.detect(orig, save_prefix=None)
    
    # ── Step 2 & 3: Lấy logic tương tự main.py ──
    # Mặc định lấy u2net nếu có, nếu không lấy orig
    if result.get('corners') is not None or result.get('u2net_doc') is not None:
        if result.get('u2net_doc') is not None:
            img_for_dewarp = result['u2net_doc']
            corners_for_dewarp = result['corners']
            anti_pinch_warped = None
            is_flat = False
            
            if corners_for_dewarp is not None and len(corners_for_dewarp) == 4:
                if result.get('u2net_mask') is not None:
                    mask_real = result['u2net_mask']
                    mask_poly = np.zeros_like(mask_real)
                    int_corners = np.array(corners_for_dewarp, dtype=np.int32)
                    cv2.fillPoly(mask_poly, [int_corners], 255)
                    intersection = np.logical_and(mask_real > 127, mask_poly > 127).sum()
                    union = np.logical_or(mask_real > 127, mask_poly > 127).sum()
                    iou = intersection / (union + 1e-6)
                    if iou > 0.94:
                        is_flat = True
            
            if result.get('u2net_mask') is not None and corners_for_dewarp is not None and len(corners_for_dewarp) == 4 and anti_pinch_warped is None:
                from step2_coons_patch import CoonsPatchDewarper
                coons_dewarper = CoonsPatchDewarper()
                img_for_dewarp = coons_dewarper.dewarp_via_contour(img_for_dewarp, result['u2net_mask'], corners_for_dewarp, save_prefix=None)
                corners_for_dewarp = None 
        else:
            img_for_dewarp = orig
            corners_for_dewarp = result['corners']
            
        warped = detector.transformer.transform(img_for_dewarp, corners_for_dewarp, save_prefix=None) \
                 if hasattr(detector.transformer, 'transform') \
                 else detector.transformer.dewarp(img_for_dewarp, corners_for_dewarp, save_prefix=None)
                 
        # Step 3
        enhanced = detector.enhancer.enhance(warped, save_prefix=None, mode="color")
        final_img = enhanced
    else:
        # Nếu không tìm thấy, trả về ảnh gốc (cho fallback)
        final_img = orig
        
    # Mã hóa ảnh trả về
    success, encoded_image = cv2.imencode('.jpg', final_img)
    if not success:
        raise HTTPException(status_code=500, detail="Error encoding final image.")
        
    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
