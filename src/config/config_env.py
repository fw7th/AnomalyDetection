"""
config.py

Configuration file for project environment.

Author: fw7th
Date: 2025-03-26
"""

import os

CLASS = 0  # Class 0 corresponds to humans in YOLO.

# Get the absolute path of the 'theft_detection' package (one level above 'src')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLY_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model
V11_PATH = os.path.join(BASE_DIR, "models", "yolov11.pt")  # Uses yolov11 object detection model.

V8_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")  # Uses yolov8 object detection model.

NAS_PATH = "yolo_nas_s"  # Uses YOLO-NAS S object detection model.

POLYGONS = os.path.join(POLY_DIR, "config.json")  # Gets the file path of the polygons.
