"""
Compare YOLOv3, SSD and RetinaNet on the same live webcam frames.

Models:
- YOLOv3u: Ultralytics implementation of YOLOv3
- SSD300 VGG16: torchvision pretrained COCO model
- RetinaNet ResNet50 FPN V2: torchvision pretrained COCO model

The program displays the three annotated frames side-by-side and prints
average inference time, FPS and approximate parameter memory at the end.
"""

import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.models.detection import (
    ssd300_vgg16, SSD300_VGG16_Weights,
    retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
)

CONFIDENCE_THRESHOLD = 0.40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# COCO class names supplied by torchvision's pretrained weights.
SSD_WEIGHTS = SSD300_VGG16_Weights.DEFAULT
RETINA_WEIGHTS = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
COCO_NAMES = SSD_WEIGHTS.meta["categories"]

print("Loading models. The first run may download pretrained weights...")

# YOLOv3u is the Ultralytics YOLOv3 model.
yolo_model = YOLO("yolov3u.pt")

ssd_model = ssd300_vgg16(weights=SSD_WEIGHTS).to(DEVICE).eval()
retina_model = retinanet_resnet50_fpn_v2(
    weights=RETINA_WEIGHTS
).to(DEVICE).eval()

def parameter_memory_mb(model):
    """Approximate memory occupied by model parameters (not the full file size)."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

def draw_torchvision_detections(frame, output, title):
    """Draw torchvision detection output on a copy of the original frame."""
    result = frame.copy()
    boxes = output["boxes"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)
        class_name = COCO_NAMES[int(label)]
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            result, f"{class_name}: {score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    cv2.putText(result, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)
    return result

def run_torchvision_model(model, frame):
    """Run a torchvision detector on a BGR OpenCV frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    with torch.inference_mode():
        output = model([tensor.to(DEVICE)])[0]
    return output

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

times = {"YOLOv3": [], "SSD": [], "RetinaNet": []}

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLOv3
        start = time.perf_counter()
        yolo_result = yolo_model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        yolo_time = time.perf_counter() - start
        times["YOLOv3"].append(yolo_time)
        yolo_view = yolo_result.plot()
        cv2.putText(yolo_view, f"YOLOv3: {1/yolo_time:.1f} FPS",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # SSD
        start = time.perf_counter()
        ssd_output = run_torchvision_model(ssd_model, frame)
        ssd_time = time.perf_counter() - start
        times["SSD"].append(ssd_time)
        ssd_view = draw_torchvision_detections(frame, ssd_output, f"SSD: {1/ssd_time:.1f} FPS")

        # RetinaNet
        start = time.perf_counter()
        retina_output = run_torchvision_model(retina_model, frame)
        retina_time = time.perf_counter() - start
        times["RetinaNet"].append(retina_time)
        retina_view = draw_torchvision_detections(
            frame, retina_output, f"RetinaNet: {1/retina_time:.1f} FPS"
        )

        # Resize to equal dimensions before joining.
        height, width = 360, 480
        views = [
            cv2.resize(yolo_view, (width, height)),
            cv2.resize(ssd_view, (width, height)),
            cv2.resize(retina_view, (width, height))
        ]
        comparison = np.hstack(views)
        cv2.imshow("YOLOv3 | SSD | RetinaNet - press Q to quit", comparison)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

print("\n========== MODEL COMPARISON ==========")
models = [
    ("YOLOv3", yolo_model.model),
    ("SSD", ssd_model),
    ("RetinaNet", retina_model)
]
for name, model in models:
    values = times[name]
    if values:
        average_time = sum(values) / len(values)
        average_fps = 1.0 / average_time if average_time else 0
        memory_mb = parameter_memory_mb(model)
        print(f"\n{name}")
        print(f"Average inference time: {average_time * 1000:.2f} ms/frame")
        print(f"Average FPS: {average_fps:.2f}")
        print(f"Approx. parameter memory: {memory_mb:.2f} MB")
    else:
        print(f"\n{name}: no frames processed.")
print("\nNote: FPS and inference time depend on hardware, input size and device.")
print("The parameter-memory value is an estimate, not the downloaded weights-file size.")
