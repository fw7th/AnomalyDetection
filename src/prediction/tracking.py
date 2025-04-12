"""
tracking.py

Provides optimized functionality for tracking and labeling of detected humans.
This module includes classes and methods for handling tracking and labeling with improved performance.

Author: fw7th
Date: 2025-04-02
"""

from src.utils.display import VideoDisplay
from src.utils.time import ClockBasedTimer
import supervision as sv
import multiprocessing
import numpy as np
import torch.cuda
import threading
import queue
import time
import cv2

# Configure supervision's label annotator with performance options
LABEL_ANNOTATOR = sv.LabelAnnotator(
    text_thickness=1,
    text_scale=0.5,
    text_padding=2
)

CORNER_ANNOTATOR = sv.BoxCornerAnnotator(
    thickness=1
)

class ObjectTracker:
    """
    Optimized multithreaded tracking system using ByteTrack.
    """

    def __init__(self, maxsize=25):
        """
        Initializes an ObjectTracker with dedicated queues for better performance.
        """
        self.display_worker = VideoDisplay()
        self.display_queue = self.display_worker.display_queue
        # Initialize tracker with better parameters
        self.timer = ClockBasedTimer()
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            from bot_sort import BoTSORT
            self.tracker = BoTSORT(
                track_high_thresh=0.6,  # Confidence threshold for new tracks
                track_low_thresh=0.1,   # Lower threshold for keeping tracks
                new_track_thresh=0.7,   # Threshold for new track initiation
                track_buffer=30,        # How long to keep lost tracks
                match_thresh=0.8,       # Matching threshold
                with_reid=True,         # Enables re-ID for better tracking
                use_cuda=True
            )
            from src.prediction.detection import ObjectDetector
            self.detector = ObjectDetector()
            self.thread_running = threading.Event()
            self.tracker_queue = queue.Queue(maxsize)
            self.worker = None
            self.detection_queue = self.detector.detection_queue
        else:
            from src.prediction.bytetrack import OpticalFlowByteTrack as OFB
            self.tracker = OFB()
            self.detection_recv = None
            self.process_running = multiprocessing.Event()
            self.processes = []

    def start(self):
        if self.use_gpu:
            self.thread_running.set()
            self.worker = threading.Thread(target=self.gpu_track, daemon=True)
            self.worker.start()

        else:
            self.process_running.set()
            for _ in range(2):
                p = multiprocessing.Process(target=self.cpu_track, args=(_,))
                p.start()
                self.processes.append(p)

    def update(self, detections, frame):
        """
        Updates the tracker with new detections and returns the tracked objects.
        
        detections: Supervision Detections object (from YOLO results)
        frame: Image frame (used for optical flow if needed)
        """
        if self.use_gpu:
            # Convert Supervision detections to BOT-SORT format
            dets = np.array([
                [x1, y1, x2, y2, score, class_id]
                for (x1, y1, x2, y2), score, class_id in zip(
                    detections.xyxy, detections.confidence, detections.class_id
                )
            ], dtype=np.float32)

            # Run BOT-SORT tracking
            tracked_objects = self.tracker.update(dets, frame)        

        return tracked_objects

    def cpu_track(self, process_idx=0):
        print(f"Starting tracking process")

        while self.process_running.is_set():
            try:
                print("Waiting for detections from pipe...")
                if not hasattr(self, 'tracker_pipe') or self.tracker_pipe is None:
                    print("ERROR: tracker_pipe not initialized!")
                    time.sleep(1)
                    continue

                if self.tracker_pipe.poll(1.0):  # Check if data available with timeout
                    frame, detections = self.tracker_pipe.recv()
                    print(f"""
                    Frame gotten from detector, ready for tracks.
                    Detections: {len(detections) if detections else 0}
                    """)
                ## frame_counter += 1

                    ## if frame_counter % len(self.processes) == process_idx:
                    try:
                        # Update tracking with new detections
                        tracked_detections = self.tracker.update_with_detections(detections, frame)
                        print("Tracked objects obtained")

                        time_in_area = self.timer.tick(tracked_detections)
                                 
                        # Process detections with tracking
                        if len(detections) > 0:
                            # Add detections visualization
                            frame_with_detections = CORNER_ANNOTATOR.annotate(
                                frame.copy(), 
                                detections=tracked_detections,
                            )

                            # Generate labels with tracking IDs and time
                            if tracked_detections.tracker_id is not None:
                                labels = [
                                    f"ID {tracker_id}, {times:.1f}s"
                                    for tracker_id, times in zip(
                                        tracked_detections.tracker_id, 
                                        time_in_area
                                    )
                                ]

                            else:
                                print("No tracked_IDs found")

                            # Create final frame with tracking labels
                            final_frame = LABEL_ANNOTATOR.annotate(
                                frame_with_detections, 
                                detections=tracked_detections, 
                                labels=labels
                            )

                        else:
                            print("Could not display IDs, might be a problem with occlusions or dark features.")
                            final_frame = frame_with_detections
                        
                        try:
                            self.display_queue.put_nowait(final_frame)

                        except queue.Full:
                            print("Tracker is slow, need more powerr!!")
                            time.sleep(0.1)
                    
                    except Exception as e:
                        print(f"Error in tracking loop: {e}")
                        time.sleep(0.1)

                else:
                    print("No data in tracker pipe yet")
                    time.sleep(0.1)

            except queue.Empty:
                print("Detection queue empty, bottleneck!")
                time.sleep(0.2)

            except Exception as e:
                import traceback
                print(f"Tracking error: {e}")
                print(traceback.format_exc())
                time.sleep(0.1)

        print("Tracking loop ended")        

    def gpu_track(self):
        while self.thread_running.is_set():
            try:
                frame, detections = self.detection_queue.get(timeout=0.2)
                print("Frame gotten from detector, ready to track.")

                try:
                    tracked_detections = self.update(detections, frame)
                    print("Tracking objects obtained.")

                    try:
                        time_in_area = self.timer.tick(tracked_detections)

                        try:
                            # Draw tracked objects
                            for track, times in zip(tracked_detections, time_in_area):
                                x1, y1, x2, y2, track_id, score, class_id = track
                                cv2.rectangle(
                                    frame, 
                                    (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                                frame = cv2.putText(
                                    frame,
                                    f"ID {int(track_id)}, {times:.1f}s", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                                )

                                try:
                                    self.display_queue.put_nowait(frame)
                                    print("Frame placed in tracking queue!")

                                except queue.Full:
                                    print("Bottleneck found for GPU tracker, need more powerrr!!")

                        except Exception as e:
                            print(f"Can't draw bounding box and annotate detections: {e}")
                            time.sleep(0.1)

                    except Exception as e:
                        print(f"Can't calculate time spent in frame for GPU Tracker: {e}")
                        time.sleep(0.1)

                except Exception as e:
                    print(f"Something wrong with GPU tracker: {e}")
                    time.sleep(0.1)

            except queue.Empty:
                print("Queue is empty, detection bottleneck!")
                time.sleep(0.1)

    def stop(self):
        """
        Stops the tracking process safely.
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
