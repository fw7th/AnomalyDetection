from ultralytics import YOLO
from src.config import config_env
import os

model = YOLO(config_env.V8_PATH)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models"
)

os.chdir(OUTPUT_DIR)

# Optimize for CPU inference
model.export(
    format="openvino",
    optimize=True,
    dynamic=True,
    batch=3,
    half=True,
    device="cpu",
    nms=True,
    opset=13,
    imgsz=(640, 640),
    save = True,
    int8=True
)

print(f"Model exported to {OUTPUT_DIR}/yolov8n.openvino_model")
