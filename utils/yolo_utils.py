import os
from ultralytics import YOLO
import numpy as np

# Get the path relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The model is usually in the project root (one level up from utils/)
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "yolov8n.pt")
model = YOLO(MODEL_PATH)


def detect_fashion_regions(image):
    results = model(image)
    detections = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                height = y2 - y1

                # 🔥 Step 3: Optimized region splitting for better footwear signal
                regions = [
                    (x1, y1, x2, y1 + int(0.25 * height)),                   # upper (shirts/tops)
                    (x1, y1 + int(0.25 * height), x2, y1 + int(0.65 * height)), # mid (torso/belts/bags)
                    (x1, y1 + int(0.65 * height), x2, y2),                   # lower (pants/shoes - bigger)
                ]

                # Labels set to 'unknown' initially
                for region in regions:
                    detections.append({
                        "label": "unknown",
                        "bbox": region
                    })

    return detections
