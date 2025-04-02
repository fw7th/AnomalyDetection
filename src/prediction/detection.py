"""
detection.py

Provides optimized functionality for multithreaded human detection.
This module includes classes and functions for handling detection, annotation, 
and multithreading while ensuring thread safety.

Author: fw7th
Date: 2025-04-02
"""

from ultralytics import YOLO
import threading
import supervision as sv
from src.config import config_env
from src.utils import preprocessing
import cv2 as cv
import torch
import time
import queue

class ObjectDetector:
    """
    A multithreaded YOLO-based human detection system.

    This class provides a real-time, thread-safe approach to detecting humans 
    in a video stream using YOLO and Supervision.
    """

    def __init__(self, source, max_queue_size=5):
        """
        Initializes an ObjectDetector instance with improved performance.
        """
        # Load model with best device selection
        self.model = YOLO(config_env.V8_PATH, task="detect", verbose=False)
        
        # Frame handling
        self.frame = None
        self.results = None
        self.detections = None
        self.annotated = None
        
        # Thread control
        self.running = False
        self.lock = threading.Lock()
        self.source = source
        
        # Add frame queue for processing
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        
        # Processing controls
        self.frame_skip = 2  # Skip frames for performance
        self.skip_count = 0
        self.last_process_time = time.time()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0

    def threaded_capture(self):
        """
        Starts the object detection process with separate capture and processing threads.
        """
        if self.source is None:
            print("Error: No video source provided.")
            return None, None

        self.cap = cv.VideoCapture(self.source)
        if not self.cap.isOpened():
            print("Error: Failed to open video source.")
            return None, None

        # Start capture and processing threads
        self.running = True
        
        # Thread for capturing frames
        self.capture_thread = threading.Thread(
            target=self._capture_frames,
            daemon=True
        )
        self.capture_thread.start()
        
        # Thread for processing frames
        self.process_thread = threading.Thread(
            target=self._process_frames,
            daemon=True
        )
        self.process_thread.start()

        return self.cap, self

    def _capture_frames(self):
        """
        Dedicated thread for frame capture to prevent blocking.
        """
        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.1)
                continue
                
            # Skip frames if queue is getting full
            if self.frame_queue.qsize() < self.frame_queue.maxsize - 1:
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    pass  # Skip this frame if queue is full
            else:
                # If queue is full, just update the counter but don't process
                self.skip_count += 1
                
            # Brief pause to prevent CPU overload
            time.sleep(0.001)
            
    def _process_frames(self):
        """
        Dedicated thread for processing frames.
        """
        # Import the zones from the JSON file
        from src.utils.general import load_zones_config

        # Load polygons from our source video
        polygons = load_zones_config(config_env.POLYGONS)
        zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,)
            )
            for polygon in polygons
        ]
        
        frame_count = 0
        
        while self.running:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(timeout=0.5)
                
                # Update FPS counter
                frame_count += 1
                current_time = time.time()
                if current_time - self.fps_timer > 5.0:
                    self.current_fps = frame_count / (current_time - self.fps_timer)
                    print(f"Detection processing rate: {self.current_fps:.1f} FPS")
                    frame_count = 0
                    self.fps_timer = current_time
                
                # Skip frames for performance based on counter
                self.skip_count += 1
                if self.skip_count % self.frame_skip != 0:
                    # Update frame but skip detection
                    with self.lock:
                        self.frame = frame.copy()
                    continue
                
                # Only preprocess and detect on selected frames
                start_time = time.time()
                
                # Selective preprocessing
                if frame.mean() < 100:  # Only for dark frames
                    preprocessed_frame = preprocessing.apply_clahe(frame)
                    preprocessed_frame = preprocessing.gammaCorrection(preprocessed_frame, 1.5)
                else:
                    preprocessed_frame = frame.copy()
                
                # Resize if needed
                h, w = preprocessed_frame.shape[:2]
                target_width = 640
                if w > target_width:
                    aspect_ratio = w / h
                    target_height = int(target_width / aspect_ratio)
                    preprocessed_frame = cv.resize(preprocessed_frame, (target_width, target_height), 
                                                  interpolation=cv.INTER_AREA)
                
                # Run detection with optimized parameters
                results = self.model(
                    preprocessed_frame,
                    classes=[0],  # Person class
                    conf=0.25,    # Higher confidence threshold for fewer false positives
                    iou=0.45,     # Lower IOU threshold for better performance
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )[0]
                
                detections = sv.Detections.from_ultralytics(results)
                
                # Apply zone filtering only if there are detections
                annotated = preprocessed_frame.copy()
                
                # Draw zones
                for idx, zone in enumerate(zones):
                    annotated = sv.draw_polygon(
                        scene=annotated,
                        polygon=zone.polygon,
                        thickness=2
                    )
                    
                # Filter detections by zones
                if len(detections) > 0:
                    for zone in zones:
                        detections = detections[zone.trigger(detections)]
                
                # Add FPS indicator
                cv.putText(
                    annotated,
                    f"FPS: {self.current_fps:.1f}",
                    (10, 30), 
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                process_time = time.time() - start_time
                
                # Store results atomically
                with self.lock:
                    self.frame = preprocessed_frame
                    self.detections = detections
                    self.annotated = annotated
                    self.results = results
                    self.last_process_time = process_time
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in detection processing: {e}")
                time.sleep(0.1)

    def get_latest_results(self):
        """
        Retrieves the latest detection results in a thread-safe manner.
        """
        with self.lock:
            return self.detections, self.annotated

    def get_performance_stats(self):
        """
        Returns the current performance statistics.
        """
        with self.lock:
            return {
                "fps": self.current_fps,
                "process_time": self.last_process_time,
                "queue_size": self.frame_queue.qsize(),
                "frame_skip": self.frame_skip
            }

    def stop(self):
        """
        Stops all threads and releases resources.
        """
        self.running = False
        
        # Wait for threads to finish
        if hasattr(self, "capture_thread"):
            self.capture_thread.join(timeout=1.0)
        if hasattr(self, "process_thread"):
            self.process_thread.join(timeout=1.0)
            
        # Release capture
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
