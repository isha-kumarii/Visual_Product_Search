import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.dirname(__file__)
    )
)

from PIL import Image
from src.fashion_detector import detect_fashion_items

# ------------------------------------------------
# Load Test Image
# ------------------------------------------------

image = Image.open(
    "test_images/01_7_additional.jpg"
).convert("RGB")

# ------------------------------------------------
# Run Detection
# ------------------------------------------------

detections = detect_fashion_items(image)

# ------------------------------------------------
# Print Results
# ------------------------------------------------

print("\nDetections:\n")

for det in detections:

    print(det)