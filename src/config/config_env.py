"""
config.py

Configuration file for project environment.

Author: fw7th
Date: 2025-03-26
"""

import os

# Get the absolute path of the 'theft_detection' package (one level above 'src')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLY_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n.pt")  # Uses yolo11n object detection model.
USE_GPU = True
CONFIDENCE_THRESHOLD = 0.5
POLYGONS = os.path.join(POLY_DIR, "config.json")  # Gets the file path of the polygons.
