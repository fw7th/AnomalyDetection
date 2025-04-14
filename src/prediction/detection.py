"""
detection.py

Provides optimized functionality for multithreaded human detection.
This module includes classes and functions for handling detection, annotation, 
and multithreading while ensuring thread safety.

Author: fw7th
Date: 2025-04-02
"""

from src.utils.general import load_zones_config  # Import the zones from the JSON file
from src.config import config_env
from ultralytics.utils.ops import LOGGER
from ultralytics import YOLO
import multiprocessing as mp
import supervision as sv
import numpy as np
import threading
import logging
import torch
import queue
import time
import os

# Silence YOLO's logger before loading the model
LOGGER.setLevel(logging.ERROR)  # Only show errors, not info messages

# Also silence the underlying libraries
os.environ['PYTHONIOENCODING'] = 'utf-8'  # Ensures proper encoding for suppressed output
os.environ['ULTRALYTICS_SILENT'] = 'True'  # Specifically for ultralytics

class ObjectDetector:
    """
    A multithreaded YOLO-based human detection system.

    This class provides a real-time, thread-safe approach to detecting humans 
    in a video stream using YOLO and Supervision.
    """

    def __init__(self, preprocessed_queue, detection_queue):
        """
        Initializes an ObjectDetector instance with improved performance.
        """
        # Load model with best device selection
        self.use_gpu = torch.cuda.is_available()

        self.detections = None
        self.results = None
        self.frame = None

        self.preprocessed_queue = preprocessed_queue
        self.detection_queue = detection_queue

        if self.use_gpu:
            self._running = threading.Event()
        else: 
            self._running = mp.Event()

    def _detect_humans(self):
        print("Starting detection process with PID:", os.getpid())

        polygons = load_zones_config(config_env.POLYGONS)
        model = YOLO(config_env.V8_PATH, task="detect", verbose=False)

        print("Model initialized!")
        zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,)
            )
            for polygon in polygons
        ]
        
        frame_count = 0
        
        while self._running.is_set():
            try:
                # Get frame from queue with timeout or pipe
                self.frame = self.preprocessed_queue.get(timeout=0.2)
                print("frame gotten from preprocessing_queue")
            
            except queue.Empty:
                print("Waiting for preprocessed frames... bottleneck!")
                time.sleep(0.1)

            except Exception as e:
                print(f"Error in detection frame extraction from preprocessing: {e}")
                time.sleep(0.1)
            
            print("We're here now")
            # Run detection on this frame
            try:
                print(f"Starting YOLO detection on frame shape: {self.frame.shape}")
                
                # First check if frame is valid
                if self.frame is None or not isinstance(self.frame, np.ndarray) or self.frame.size == 0:
                    print("ERROR: Invalid frame for detection")
                    break

                if model is False or None:
                    print("Error: Model not initialized")
                    break
                
                try:
                    print("YOLO-based detection should start")
                    self.results = model(
                        self.frame,
                        classes=[0],  # Person class
                        conf=0.25,    # Higher confidence threshold for fewer false positives
                        iou=0.45,     # Lower IOU threshold for better performance
                        device="cuda" if self.use_gpu else "cpu"
                    )[0]
                    print(f"YOLO detection completed: {type(self.results)}")

                except Exception as e:
                    import traceback
                    print(f"""
                    ===== YOLO DETECTION ERROR =====
                    Error type: {type(e).__name__}
                    Error message: {str(e)}
                    ------- Stack trace: -------
                    {traceback.format_exc()}
                    ===============================
                    """)
                    continue
                
                print("Converting detection results to Supervision format")

            except Exception as e:
                import traceback
                print(f"Outer detection loop error: {e}")
                print(traceback.format_exc())
                time.sleep(0.1)

            try:
                self.detections = sv.Detections.from_ultralytics(self.results)

            except Exception as e:
                print("ERROR in Supervison conversion!")
                import traceback
                print(f"Error creating detections: {e}")
                print(traceback.format_exc())
                continue
        
            print("Time to draw zone on frame")
            if self.frame is not None:
                # Draw zones
                for idx, zone in enumerate(zones):
                    self.frame = sv.draw_polygon(
                        scene=self.frame,
                        polygon=zone.polygon,
                        thickness=2
                    )
                   
                print("Zone drawn on frame")
                # Filter detections by zones
                if self.detections:
                    for zone in zones:
                        self.detections = self.detections[zone.trigger(self.detections)]
               
                print(f"self.detections: {type(self.detections)}")
                print(f"self.detections: {type(self.frame)}")

                try:
                    self.detection_queue.put((self.frame, self.detections))
                    print("frame and detections sent to tracker")
                    
                    # Force garbage colection every few frames
                    frame_count += 1
                    if frame_count % 5 == 0:
                        import gc 
                        gc.collect()
                        torch.cuda.empty_cache if self.use_gpu else None
                        print(f"Memory cleaned after {frame_count} frames")

                except queue.Full:
                    print("Detection or frame queue is full, need more powerrr!!")
                    time.sleep(0.2)

                except Exception as e:
                    print(f"Error: Failed to send the detection and frame to pipe or queue")

                else:
                    print("Detections are null or frame is empty")

        print("Detection loop ended")
