"""
tracking.py

Provides optimized functionality for tracking and labeling of detected humans.
This module includes classes and methods for handling tracking and labeling with improved performance.

Author: fw7th
Date: 2025-04-02
"""

import supervision as sv
import time
import threading
import queue
import cv2 as cv
from src.utils.time import ClockBasedTimer
from src.utils.general import CORNER_ANNOTATOR

# Configure supervision's label annotator with performance options
LABEL_ANNOTATOR = sv.LabelAnnotator(
    text_thickness=1,
    text_scale=0.5,
    text_padding=2
)

class ObjectTracker:
    """
    Optimized multithreaded tracking system using ByteTrack.
    """

    def __init__(self, input_queue_size=10):
        """
        Initializes an ObjectTracker with dedicated queues for better performance.
        """
        # Initialize tracker with better parameters
        self.tracker = sv.ByteTrack()
        
        self.timer = ClockBasedTimer()
        self.labeled_frames = None
        self.is_running = False
        self.tracking_lock = threading.Lock()
        self.error_state = False
        
        # Input queue for receiving detections to track
        self.input_queue = queue.Queue(maxsize=input_queue_size)
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
    
    def start_tracking(self):
        """Start the tracking thread with dedicated processing."""
        try:
            print("Initializing tracker...")
            
            # Start tracking thread
            self.is_running = True
            self.tracking_thread = threading.Thread(
                target=self._tracking_loop,
                daemon=True
            )
            self.tracking_thread.start()
            return True

        except Exception as e:
            print(f"Error starting tracking: {e}")
            self.error_state = True
            return False

    def _tracking_loop(self):
        """Internal method containing the tracking loop logic optimized for performance."""
        frame_count = 0
        
        while self.is_running:
            try:
                # Get detection data from queue with timeout
                try:
                    detection_data = self.input_queue.get(timeout=0.5)
                    detections, frame = detection_data
                except queue.Empty:
                    # Nothing to process
                    time.sleep(0.01)
                    continue
                
                # Skip processing if no detections or frame
                if detections is None or frame is None:
                    continue

                # Update performance counter
                frame_count += 1
                current_time = time.time()
                if current_time - self.fps_timer > 5.0:
                    self.current_fps = frame_count / (current_time - self.fps_timer)
                    print(f"Tracking processing rate: {self.current_fps:.1f} FPS")
                    frame_count = 0
                    self.fps_timer = current_time

                # Add frame counter to the image
                h, w = frame.shape[:2]
                cv.putText(
                    frame,
                    f"Track FPS: {self.current_fps:.1f}",
                    (10, 60),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1, 
                    (0, 255, 255),
                    2
                )
                
                try:
                    # Process detections with tracking
                    if len(detections) > 0:
                        # Add detections visualization
                        frame_with_detections = CORNER_ANNOTATOR.annotate(
                            frame.copy(), 
                            detections=detections
                        )
                        
                        # Update tracking with new detections
                        tracked_detections = self.tracker.update_with_detections(detections)
                        time_in_area = self.timer.tick(tracked_detections)
                        
                        # Generate labels with tracking IDs and time
                        if len(tracked_detections.tracker_id) > 0:
                            labels = [
                                f"#{tracker_id} {times:.1f}s"
                                for tracker_id, times in zip(
                                    tracked_detections.tracker_id, 
                                    time_in_area
                                )
                            ]
                            
                            # Create final frame with tracking labels
                            final_frame = LABEL_ANNOTATOR.annotate(
                                frame_with_detections, 
                                detections=tracked_detections, 
                                labels=labels
                            )
                        else:
                            final_frame = frame_with_detections
                    else:
                        # No detections to track
                        final_frame = frame.copy()
                        
                    # Store the frame safely
                    with self.tracking_lock:
                        self.labeled_frames = final_frame
                    
                except Exception as e:
                    print(f"Error in tracking processing: {e}")
                    with self.tracking_lock:
                        self.labeled_frames = frame  # Use original frame on error
                
            except Exception as e:
                print(f"Error in tracking loop: {e}")
                time.sleep(0.1)
        
        print("Tracking loop ended")
        
    def add_detection_for_tracking(self, detections, frame):
        """
        Add detections to the tracking queue.
        
        Returns True if successfully added to queue, False otherwise.
        """
        if detections is None or frame is None:
            return False

        try:
            # Add to queue with timeout to prevent blocking
            self.input_queue.put((detections, frame), timeout=0.1)
            return True
        except queue.Full:
            # Queue is full, skip this detection
            return False
                        
    def return_frames(self):
        """
        Thread-safe method to get the latest labeled frames.
        """
        with self.tracking_lock:
            return self.labeled_frames

    def get_error_state(self):
        """
        Check if the tracker encountered any fatal errors.
        """
        return self.error_state

    def stop(self):
        """
        Stops the tracking process safely.
        """
        print("Stopping tracking...")
        self.is_running = False
        
        # Wait for thread to finish
        if hasattr(self, "tracking_thread"):
            self.tracking_thread.join(timeout=1.0)
