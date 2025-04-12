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
from src.prediction.tracking import ObjectTracker
from ultralytics import YOLO
import supervision as sv
import multiprocessing as mp
import numpy as np
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

    def __init__(self, maxsize=25):
        """
        Initializes an ObjectDetector instance with improved performance.
        """
        # Load model with best device selection
        self.model = YOLO(config_env.V8_PATH, task="detect", verbose=False)
        self.tracker = ObjectTracker()  
        self.use_gpu = torch.cuda.is_available()

        self.frame = None
        self.detections = None
        self.results = None

        if self.use_gpu:
            self.thread_running = threading.Event()
            self.detection_queue = queue.Queue(maxsize)
            self.thread_worker = None

        else: 
            self.preprocess_recv = None  # Will be set by FrameProcessor_
            self.tracker.detection_recv, self.detection_send= mp.Pipe(duplex=False)
            self.process_running = mp.Event()
            self.detection_queue = mp.Queue(maxsize)

    def start(self):
        if self.use_gpu:
            self.thread_running.set()
            self.thread_worker = threading.Thread(target=self._detect_humans, daemon=True)
            self.thread_worker.start()

        else:
            self.process_running.set()
            for _ in range(2):
                p = mp.Process(target=self._detect_humans, args=(_,))
                p.start()
                self.processes.append(p)

    def _detect_humans(self):
        # Only process every Nth frame based on process_idx

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
                # Get frame from queue with timeout or pipe
                if self.use_gpu:
                    self.frame = self.preprocessed_queue.get(timeout=0.2)
                    print("frame gotten from preprocessing_queue")
                else:
                    self.frame = self.preprocess_recv.recv()
                    print("frame gotten from preprocessing_pipe")
            
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

                if self.model is False or None:
                    print("Error: Model not initialized")
                    break
                
                try:
                    self.results = self.model(
                        self.frame,
                        classes=[0],  # Person class
                        conf=0.25,    # Higher confidence threshold for fewer false positives
                        iou=0.45,     # Lower IOU threshold for better performance
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    )[0]
                    print(f"YOLO detection completed: {type(self.results)}")
                    print(f"results.boxes: {self.results.boxes}")
                    print(f"results.boxes.xyxy: {getattr(self.results.boxes, 'xyxy', 'no xyxy')}")
                    print(f"results shape: {getattr(self.results, 'orig_shape', 'no shape')}")


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
                print(f"Detections created: {len(self.detections)} objects")

            except Exception as e:
                print("ERROR in Supervison conversion!")
                import traceback
                print(f"Error creating detections: {e}")
                print(traceback.format_exc())
                continue
        
            # More detailed debugging
            print(f"""
            Detection pipeline status:
            - Frame shape: {self.frame.shape}
            - Results type: {type(self.results)}
            - Detections count: {len(self.detections) if self.detections else 0}
            - Zone count: {len(zones)}
            - Next step: Processing zones
            """)

            # Draw zones
            for idx, zone in enumerate(zones):
                self.frame = sv.draw_polygon(
                    scene=self.frame,
                    polygon=zone.polygon,
                    thickness=2
                )
                
            # Filter detections by zones
            if len(self.detections) > 0:
                for zone in zones:
                    self.detections = self.detections[zone.trigger(self.detections)]
            
            if len(self.frame) > 0 and self.detections:
                try:
                    if self.use_gpu:
                        self.detection_queue.put((self.frame, self.detections))
                        if self.detection_queue.full():
                            discarded = self.detection_queue.get_nowait()
                    else:
                        self.detection_send.send((self.frame, self.detections))
                        print("frame and detections sent to tracker_pipe")

                except queue.Full:
                    print("Detection or frame queue is full, need more powerrr!!")
                    time.sleep(0.2)


            else:
                print("Detections are null or frame is empty")

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

