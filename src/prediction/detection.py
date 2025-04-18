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
# import threading
import cv2 as cv
import logging
import torch
import queue
import time
import os

# Set PyTorch thread count to optimize for 2 cores
torch.set_num_threads(2)

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

        self.use_gpu = torch.cuda.is_available()
        self.last_log_time = time.time()
        self.frame_count = 0

        self.detections = None
        self.results = None
        self.frame = None
        self.buffer = []

        # Initialize annotator as instance variable
        self.corner_annotator = sv.BoxCornerAnnotator(thickness=1)

        self.model = None
        self.zones = self._initialize_zones()

        self.preprocessed_queue = preprocessed_queue
        self.detection_queue = detection_queue
        
        """
        if self.use_gpu:
            self._running = threading.Event()
        """
        self._running = mp.Event()
        self.mutex = mp.Manager().RLock()

    def _initialize_model(self):
        # Silence YOLO's logger before loading the model
        LOGGER.setLevel(logging.ERROR)  # Only show errors, not info messages

        # Also silence the underlying libraries
        os.environ['PYTHONIOENCODING'] = 'utf-8'  # Ensures proper encoding for suppressed output
        os.environ['ULTRALYTICS_SILENT'] = 'True'  # Specifically for ultralytics
        
        try:
            # Load model with best device selection
            model = YOLO(config_env.V8_PATH)
            """ 
            if self.use_gpu:
                # Optimize for GPU inference
                model.to('cuda')
                if hasattr(model, 'model'):
                    # Only apply half precision if GPU supports it
                    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
                        model.model.half()
            """
            # Optimize for CPU inference
            model.export(
                format="onnx",
                optimize=True,
                dynamic=True,
                batch=1,
                half=True,
                device="cpu",
                nms=True,
                opset=13
            )

            return model

        except Exception as e:
            print(f"Error initializing model: {e}")
            return None

    def _initialize_zones(self):
        """Initialize detection zones once"""

        try:
            polygons = load_zones_config(config_env.POLYGONS)
            return [
                sv.PolygonZone(
                    polygon=polygon,
                    triggering_anchors=(sv.Position.CENTER,)
                )
                for polygon in polygons
            ]
        except Exception as e:
            print(f"Error initializing zones: {e}")
            return []

    def detect_single_frame(self):
        try:
            # Get frame from queue with timeout
            self.mutex.acquire()
            self.frame = self.preprocessed_queue.get(timeout=0.001)
            self.mutex.release()
        
        except queue.Empty:
            time.sleep(0.01)
            print("Waiting for preprocessed frames... bottleneck!")
            return

        except Exception as e:
            print(f"Error in detection frame extraction from preprocessing: {e}")
            time.sleep(0.01)
            return
        
        # Run detection on this frame
        try:
            # First check if frame is valid
            if self.frame is None or not isinstance(self.frame, np.ndarray) or self.frame.size == 0:
                print("ERROR: Invalid frame for detection")
                return
                
            # Check model initialization
            if self.model is None:
                print("Model is not initialized")
                self.model = self._initialize_model()
                if self.model is None:
                    return

            try:
                self.results = self.model(
                    self.frame,
                    classes=[0],  # Person class
                    conf=0.35,    # Higher confidence threshold for fewer false positives
                    iou=0.45,     # Lower IOU threshold for better performance
                    device="cuda" if self.use_gpu else "cpu",
                    augment=False,
                    agnostic_nms=True,
                    task="detect",
                    verbose=False
                )[0]

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
                return 
            
        except Exception as e:
            import traceback
            print(f"Outer detection loop error: {e}")
            print(traceback.format_exc())
            time.sleep(0.1)
            return

        try:
            # Convert YOLO results to Supervision detections
            self.detections = sv.Detections.from_ultralytics(self.results)
        except Exception as e:
            print("ERROR in Supervision conversion!")
            import traceback
            print(f"Error creating detections: {e}")
            print(traceback.format_exc())
            return
    
        # Process valid frame with detections
        if self.detections is not None:
            processed_frame = self.frame.copy()
            
            # Process zones if they exist
            if self.zones:
                # Draw all zones
                for idx, zone in enumerate(self.zones):
                    processed_frame = sv.draw_polygon(
                        scene=processed_frame,
                        polygon=zone.polygon,
                        thickness=2
                    )
                
                # Create a mask for combined zone filtering
                combined_mask = np.zeros(len(self.detections), dtype=bool)
                if len(self.detections) > 0:
                    for zone in self.zones:
                        # Use OR to combine results from all zones
                        zone_mask = zone.trigger(self.detections)
                        combined_mask = np.logical_or(combined_mask, zone_mask)
                    
                    # Apply the combined filter
                    if np.any(combined_mask):
                        self.detections = self.detections[combined_mask]
                    else:
                        # Keep original detections if no zones triggered
                        pass
            
            # Add detections visualization if any detections remain
            if len(self.detections) > 0:
                processed_frame = self.corner_annotator.annotate(
                    scene=processed_frame, 
                    detections=self.detections
                )
            
            # Log FPS periodically using the class frame counter
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_log_time >= 4.0:  # Log every 4 seconds
                elapsed = current_time - self.last_log_time
                fps = self.frame_count / elapsed
                print(f"Inference at {fps:.2f} FPS")
                self.frame_count = 0
                self.last_log_time = current_time

            time_now = time.time()
            if self.detections and int(time_now * 2) % 2 == 0:
                processed_frame = cv.rectangle(processed_frame, (0, 0), (640, 360), (0, 0, 200), 30)
                processed_frame = cv.putText(
                    processed_frame,
                    "Threat Detected",
                    (200, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
            try:
                # Send the processed frame to the output queue
                self.mutex.acquire()
                self.detection_queue.put(processed_frame, timeout=0.001)
                self.mutex.release()
                
                # Force garbage collection periodically
                if self.frame_count % 30 == 0:  # Less frequent GC
                    import gc 
                    gc.collect()
                    if self.use_gpu:
                        torch.cuda.empty_cache()
                    print(f"Memory cleaned after {self.frame_count} frames")

            except queue.Full:
                print("Detection queue is full, dropping frame")
                time.sleep(0.01)

            except Exception as e:
                print(f"Error: Failed to send processed frame to queue: {e}")
                return
