"""
Custom car detection example from the lab's YOLO implementation section.

IMPORTANT:
1. Train a YOLO model on the provided custom street/car dataset first.
2. Put the resulting best.pt file in this folder (or update MODEL_PATH).
3. This script uses that trained model; COCO pretrained weights alone are
   not a substitute for a custom car dataset if the lab requires custom classes.
"""

import cv2
from ultralytics import YOLO

MODEL_PATH = "best.pt"  # Replace with the path to your trained custom weights.
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.predict(frame, conf=0.40, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Custom Car Detection - press Q to quit", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
