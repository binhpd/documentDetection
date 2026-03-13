from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import cv2

image_path = "../image/fold/0004.jpg"
doc = DocumentFile.from_images(image_path)
model = ocr_predictor(pretrained=True)
result = model(doc)

for page in result.pages:
    print(f"Page dimensions: {page.dimensions}")
    if len(page.blocks) > 0:
        xmin = min([b.geometry[0][0] for b in page.blocks])
        ymin = min([b.geometry[0][1] for b in page.blocks])
        xmax = max([b.geometry[1][0] for b in page.blocks])
        ymax = max([b.geometry[1][1] for b in page.blocks])
        print(f"Text content bounding box (relative): ({xmin}, {ymin}) to ({xmax}, {ymax})")
    else:
        print("No text found")
