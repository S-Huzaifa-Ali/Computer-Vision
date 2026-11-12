"""
Multi-class safety monitor: person, bike/motorcycle and helmet.

Train a custom YOLO model with class labels that include at least:
    person, motorcycle (or bike), helmet

Update MODEL_PATH and the class names to match your dataset.
The compliance rule below is a simple demonstration: if a motorcycle is
detected and no helmet is detected in the frame, show a warning.

For a real system, associate each helmet with a specific rider/person;
frame-level presence/absence is not reliable enough for real enforcement.
"""

import cv2
from ultralytics import YOLO

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Change these names if your dataset uses different class labels.
HELMET_NAMES = {"helmet", "hardhat"}
BIKE_NAMES = {"motorcycle", "motorbike", "bike"}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model.predict(frame, conf=0.40, verbose=False)[0]
    annotated = result.plot()

    detected_names = set()
    if result.boxes is not None:
        for class_id in result.boxes.cls.tolist():
            detected_names.add(str(model.names[int(class_id)]).lower())

    bike_detected = bool(detected_names & BIKE_NAMES)
    helmet_detected = bool(detected_names & HELMET_NAMES)

    if bike_detected and not helmet_detected:
        message = "WARNING: Bike detected - no helmet detected"
        color = (0, 0, 255)
    elif bike_detected and helmet_detected:
        message = "Bike and helmet detected"
        color = (0, 255, 0)
    else:
        message = "Monitoring..."
        color = (255, 255, 0)

    cv2.rectangle(annotated, (5, 5), (annotated.shape[1] - 5, 45), (0, 0, 0), -1)
    cv2.putText(annotated, message, (15, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    cv2.imshow("Helmet Compliance Monitor - press Q to quit", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
