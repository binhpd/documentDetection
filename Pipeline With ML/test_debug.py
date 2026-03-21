import glob
import sys
from page_dewarp.image import WarpedImage
from page_dewarp.options import Config

config = Config()
if hasattr(config, "debug_level"):
    config.debug_level = 2
if hasattr(config, "DEBUG_LEVEL"):
    config.DEBUG_LEVEL = 2

try:
    WarpedImage("01_u2net_extracted_curved_doc.jpg", config=config)
except Exception as e:
    print("Error:", e)

print("Debug PNGs:", glob.glob("*debug*.png"))
print("All Output:", glob.glob("*u2net*.png"))
