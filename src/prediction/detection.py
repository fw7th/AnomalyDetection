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
from ultralytics import YOLO
import supervision as sv
import multiprocessing
import threading
import torch
import queue
import time

class ObjectDetector:
    """
    A multithreaded YOLO-based human detection system.

    This class provides a real-time, thread-safe approach to detecting humans 
    in a video stream using YOLO and Supervision.
    """

    def __init__(self, preprocessed_queue, maxsize=25):
        """
        Initializes an ObjectDetector instance with improved performance.
        """
        # Load model with best device selection
        self.model = YOLO(config_env.V8_PATH, task="detect", verbose=False)
        self.preprocessed_queue = preprocessed_queue
        
        self.use_gpu = torch.cuda.is_available()

        # Frame handling
        self.frame = None
        self.results = None
        self.detections = None
       
        if self.use_gpu:
            self.thread_running = threading.Event()
            self.detection_queue = queue.Queue(maxsize)
            self.thread_worker = None

        else: 
            self.process_running = multiprocessing.Event()
            self.detection_queue = multiprocessing.Queue(maxsize)
            self.processes = []

    def start(self):
        if self.use_gpu:
            self.thread_running.set()
            self.thread_worker = threading.Thread(target=self._detect_humans, daemon=True)
            self.thread_worker.start()

        else:
            self.process_running.set()
            for _ in range(2):
                p = multiprocessing.Process(target=self._detect_humans, args=(_,))
                p.start()
                self.processes.append(p)

    def _detect_humans(self, process_idx=0):
        # Only process every Nth frame based on process_idx
        frame_counter = 0

        # Load polygons from our source video
        polygons = load_zones_config(config_env.POLYGONS)
        zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,)
            )
            for polygon in polygons
        ]

        while (self.thread_running.is_set() if self.use_gpu else self.process_running.is_set()):
            try:
                # Get frame from queue with timeout
                frame = self.preprocessed_queue.get(timeout=0.3)
                frame_counter += 1
                
                # Process only frames that match this process's index
                if not self.use_gpu and frame_counter % len(self.processes) != process_idx:
                    # Run detection on this frame
                    results = self.model(
                        frame,
                        classes=[0],  # Person class
                        conf=0.25,    # Higher confidence threshold for fewer false positives
                        iou=0.45,     # Lower IOU threshold for better performance
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    )[0]

                    detections = sv.Detections.from_ultralytics(results)

                    # Draw zones
                    for idx, zone in enumerate(zones):
                        frame = sv.draw_polygon(
                            scene=frame,
                            polygon=zone.polygon,
                            thickness=2
                        )
                        
                    # Filter detections by zones
                    if len(detections) > 0:
                        for zone in zones:
                            detections = detections[zone.trigger(detections)]
                    
                    if frame.size and detections:
                        try:
                            self.detection_queue.put((frame, detections))

                            if self.detection_queue.full():
                                discarded = self.detection_queue.get_nowait()

                        except queue.Full:
                            print("Detection or frame queue is full, need more powerrr!!")
                            time.sleep(0.2)
                    else:
                        print("Detections are null or frame is empty")
                    
            except queue.Empty:
                print("Waiting for preprocessed frames... bottleneck!")
                time.sleep(0.1)

            except Exception as e:
                print(f"Error in detection processing: {e}")
                time.sleep(0.1)

        print("Detection loop ended")

    def stop(self):
        """
        Stops all threads and releases resources.
        """
        if self.use_gpu:
            self.thread_running.clear()
            if self.worker and self.worker.is_alive():
                self.worker.join(timeout=2)

        else:
            self.process_running.clear()
            for p in self.processes:
                if p and p.is_alive():
                    p.join(timeout=2)
                    if p.is_alive():
                        p.terminate()

