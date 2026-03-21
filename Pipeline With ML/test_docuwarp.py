import cv2
import sys
import numpy as np
try:
    from docuwarp import init_model, unwarp
    print("Successfully imported docuwarp!")
    
    # Load model
    print("Loading UVDoc model...")
    model = init_model()
    
    # Load image
    image_path = "../perspective/0009.jpg"
    print(f"Loading image {image_path}")
    image = cv2.imread(image_path)
    
    # Run unwarp
    print("Running unwarp...")
    unwarped_image = unwarp(model, image)
    
    # Save output
    cv2.imwrite("test_dewarped_0009.jpg", unwarped_image)
    print("Saved test_dewarped_0009.jpg")
    
except Exception as e:
    print(f"Error: {e}")
