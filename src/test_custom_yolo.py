from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# ------------------------------------------------
# Load custom model
# ------------------------------------------------

model = YOLO(
    "models/fashion_yolo_best.pt"
)

print("Custom YOLO loaded successfully")

# ------------------------------------------------
# Test image
# ------------------------------------------------

image_path = (
    "data/DeepFashion/"
    "Img/WOMEN/Skirts/"
    "id_00000003/01_1_front.jpg"
)

# ------------------------------------------------
# Run inference
# ------------------------------------------------

results = model(image_path)

# ------------------------------------------------
# Print detections
# ------------------------------------------------

for box in results[0].boxes:

    class_id = int(box.cls[0])

    confidence = float(box.conf[0])

    class_name = (
        model.names[class_id]
    )

    print(
        f"{class_name} "
        f"({confidence:.2f})"
    )

# ------------------------------------------------
# Visualize
# ------------------------------------------------

annotated = results[0].plot()

plt.figure(figsize=(8,8))

plt.imshow(annotated)

plt.axis("off")

plt.show()