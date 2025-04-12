from src.config import config_env
from ultralytics import YOLO
import torch

image = "/home/fw7th/Pictures/me.jpg"

model = YOLO(config_env.V8_PATH, task="detect", verbose=False)
results = model(
        image,
        classes=[0],  # Person class
        conf=0.25,    # Higher confidence threshold for fewer false positives
        iou=0.45,     # Lower IOU threshold for better performance
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )[0]
print(f"YOLO detection completed: {type(results)}")

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # 
