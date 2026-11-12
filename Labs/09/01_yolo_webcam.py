import cv2
from ultralytics import YOLO
import time

# YOLOv8 nano is used because it is lightweight and suitable for webcam use.
# The weights are downloaded automatically the first time the script runs.
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions or camera index.")

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read a frame from the webcam.")
        break

    start = time.perf_counter()
    results = model.predict(frame, conf=0.40, verbose=False)
    inference_time = time.perf_counter() - start

    # Ultralytics draws bounding boxes, class names, and confidence scores.add
    annotated_frame = results[0].plot()

    fps = 1.0 / inference_time if inference_time > 0 else 0
    cv2.putText(
        annotated_frame,
        f"Inference: {inference_time * 1000:.1f} ms | FPS: {fps:.1f}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
    )

    cv2.imshow("YOLO Live Object Detection - press Q to quit", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
