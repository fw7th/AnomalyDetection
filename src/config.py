import os

# Get the absolute path of the 'theft_detection' package (one level above 'src')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the path to the model
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n.pt")
USE_GPU = True
CONFIDENCE_THRESHOLD = 0.5
