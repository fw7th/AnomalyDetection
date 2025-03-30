"""
tracking.py

Provides functionality for tracking and labeling of detected humans.
This module includes classes and methods for handling tracking and labeling of tracked objects.

Author: fw7th
Date: 2025-03-25
"""

import supervision as sv
from src.prediction.detection import ObjectDetector
import time
import threading
from src.utils.time import ClockBasedTimer

LABEL_ANNOTATOR = sv.LabelAnnotator()  # Supervision's label annotator

class ObjectTracker:
    """
    Multithreaded tracking system.

    Uses ByteTrack as the tracking algorithm. The implementation is based on the Supervision library.

    Attributes
    ----------
    tracker : sv.ByteTrack
        ByteTrack tracking model from Supervision.
    detector : ObjectDetector
        Instance of the ObjectDetector class for running detections.
    timer : ClockBasedTimer
        Timer instance used to track how long each detected object stays in frame.
    labeled_frames : numpy.ndarray or None
        Stores the latest annotated frame with tracking labels.
    is_running : bool
        Indicates whether the tracking process is currently active.
    tracking_lock : threading.Lock
        Ensures thread safety when accessing shared attributes.
    error_state : bool
        Tracks whether a fatal error has occurred during runtime.
    source : int or str
        Source of the stream file.
    """

    def __init__(self, source):
        """
        Initializes an ObjectTracker instance.
        """
        self.tracker = sv.ByteTrack()
        self.timer = ClockBasedTimer()
        self.labeled_frames = None
        self.is_running = False
        self.tracking_lock = threading.Lock()  # Add a lock for thread safety
        self.error_state = False  # Track if we've encountered fatal errors
        self.source = source
        self.detector = ObjectDetector(self.source)

    def get_tracked_objects(self):
        """
        Starts tracking objects in the video source.
        This method is designed to run in its own thread.
        Labels detections with their IDs and time spent in frame.
        Raises
        ------
        ValueError
            If no video source is provided.
        RuntimeError
            If the detector fails to initialize.
        """
        if self.source is None:
            raise ValueError("No video source provided. Please specify a valid source.")

        try:
            # Initialize the detector
            cap, detector = self.detector.threaded_capture()
            if cap is None:
                raise RuntimeError("Failed to initialize detection. Ensure the video source is accessible.")

            print("Tracker initialized successfully. Starting tracking loop...")

            # Main tracking loop
            self.is_running = True
            frame_count = 0
            last_update_time = time.time()

            while cap.isOpened() and self.is_running:
                current_time = time.time()
                detections, annotated, results = detector.get_latest_results()

                if annotated is None:
                    # Adaptive waiting
                    time.sleep(min(0.1, max(0.01, (1/30) - (time.time() - current_time))))
                    continue

                try:
                    if detections is not None and results is not None:
                        # Update tracking
                        tracked_detections = self.tracker.update_with_detections(detections)
                        time_in_area = self.timer.tick(tracked_detections)

                        with self.tracking_lock:
                            if len(tracked_detections.tracker_id) > 0:
                                labels = [
                                    f"#{tracker_id} {times:.1f}s"
                                    for tracker_id, times in zip(tracked_detections.tracker_id, time_in_area)
                                ]
                                self.labeled_frames = LABEL_ANNOTATOR.annotate(
                                    annotated.copy(), 
                                    detections=tracked_detections, 
                                    labels=labels
                                )
                            else:
                                self.labeled_frames = annotated.copy()
                    else:
                        with self.tracking_lock:
                            self.labeled_frames = annotated.copy()

                    frame_count += 1
                    if time.time() - last_update_time > 5.0:
                        fps = frame_count / (time.time() - last_update_time)
                        print(f"Tracking processing rate: {fps:.1f} FPS")
                        frame_count = 0
                        last_update_time = time.time()

                    time.sleep(max(0.001, (1/30) - (time.time() - current_time)))

                except Exception as e:
                    print(f"Error during tracking: {e}")
                    time.sleep(0.1)

        except Exception as e:
            print(f"Fatal tracking error: {e}")
            self.error_state = True
        finally:
            self.cleanup()
            print("Tracking thread terminated")

    def return_frames(self):
        """
        Thread-safe method to get the latest labeled frames.

        Returns
        -------
        numpy.ndarray or None
            The latest labeled frame from the tracking system.
            Returns None if no frames have been processed yet.
        """
        with self.tracking_lock:
            return self.labeled_frames

    def get_error_state(self):
        """
        Check if the tracker encountered any fatal errors.

        Returns
        -------
        bool
            True if a fatal error occurred, otherwise False.

        Notes
        -----
        Common error causes:
        - Failure in `detector.get_latest_results()`
        - Unexpected exceptions during tracking updates
        - Issues with the video source
        """
        return self.error_state

    def stop(self):
        """
        Stops the tracking process safely.
        """
        print("Stopping tracking...")
        self.is_running = False

    def cleanup(self):
        """
        Ensures all resources are released properly and stops the detector.

        Notes
        -----
        - Calls `detector.stop()` to terminate the detection thread.
        - Ensures tracking stops even if an error occurred.
        """
        if self.detector.running:
            try:
                self.detector.stop()
                print("Detection thread stopped successfully.")
            except Exception as e:
                print(f"Error stopping detection thread: {e}")
