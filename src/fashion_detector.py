from ultralytics import YOLO
import numpy as np
from PIL import Image

# ------------------------------------------------
# Load Fashion YOLO
# ------------------------------------------------

model = YOLO(
    "models/fashion_yolo/best.pt"
)

# ------------------------------------------------
# Detect Fashion Items
# ------------------------------------------------

def detect_fashion_items(image):

    image_np = np.array(image)

    results = model(image_np)

    detections = []

    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        class_name = model.names[cls_id]

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        # ------------------------------------------------
        # Crop detected region
        # ------------------------------------------------

        crop = image_np[
            y1:y2,
            x1:x2
        ]

        crop_pil = Image.fromarray(crop)

        detections.append({

            "class": class_name,

            "bbox": (
                x1,
                y1,
                x2,
                y2
            ),

            "crop": crop_pil
        })

    return detections