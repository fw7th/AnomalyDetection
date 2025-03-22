import supervision as sv
from src.prediction.detection import ObjectDetector
import time
import threading
from src.utils.time import ClockBasedTimer

LABEL_ANNOTATOR = sv.LabelAnnotator()

class Tracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.detector = ObjectDetector()
        self.timer = ClockBasedTimer()
        self.labeled_frames = None
        self.is_running = False
        self.tracking_lock = threading.Lock()  # Add a lock for thread safety
        self.error_state = False  # Track if we've encountered fatal errors


    def get_tracked_objects(self, source=None):
        """
        Start tracking objects in the video source.
        This method is designed to run in its own thread.
        """
        if source is None:
            print("Error: No video source provided.")
            self.error_state = True
            return

        try:
            # Initialize the detector
            cap, detector = self.detector.threaded_capture(source)
            if cap is None:
                print("Error: Failed to initialize detection.")
                self.error_state = True
                return

            print("Tracker initialized successfully. Starting tracking loop...")
            
            # Main tracking loop
            self.is_running = True
            frame_count = 0
            last_update_time = time.time()
            
            while cap.isOpened() and self.is_running:
                # Get latest detection results from the detector thread
                current_time = time.time()
                detections, annotated, results = detector.get_latest_results()
                
                # Skip processing if we don't have a frame yet
                if annotated is None:
                    # Don't sleep for a fixed time - use adaptive waiting
                    wait_time = min(0.1, max(0.01, (1/30) - (time.time() - current_time)))
                    time.sleep(wait_time)
                    continue
                
                # Process frame if available
                try:
                    if detections is not None and results is not None:
                        # Update tracking
                        tracked_detections = self.tracker.update_with_detections(detections)
                        time_in_area = self.timer.tick(tracked_detections)
                        
                        # Thread-safe update of the labeled frames
                        with self.tracking_lock:
                            if len(tracked_detections.tracker_id) > 0:
                                # Create labels for tracked objects
                                labels = [
                                    f"#{tracker_id} {times/10:.1f}s"  # Format to 1 decimal place
                                    for tracker_id, times in zip(tracked_detections.tracker_id, time_in_area)
                                ]
                                self.labeled_frames = LABEL_ANNOTATOR.annotate(
                                    annotated.copy(), 
                                    detections=tracked_detections, 
                                    labels=labels
                                )
                            else:
                                # If no tracked objects, just show the annotated frame
                                self.labeled_frames = annotated.copy()
                    else:
                        # If no detections, show the raw annotated frame
                        with self.tracking_lock:
                            self.labeled_frames = annotated.copy()
                    
                    # Count processed frames for performance monitoring
                    frame_count += 1
                    if time.time() - last_update_time > 5.0:
                        fps = frame_count / (time.time() - last_update_time)
                        print(f"Tracking processing rate: {fps:.1f} FPS")
                        frame_count = 0
                        last_update_time = time.time()
                    
                    # Adaptive sleep to prevent CPU overload
                    # Sleep less if processing is slow, more if it's fast
                    processing_time = time.time() - current_time
                    target_frame_time = 1/30  # Target 30 FPS
                    sleep_time = max(0.001, target_frame_time - processing_time)
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    print(f"Error during tracking: {e}")
                    # Continue instead of breaking - try to recover
                    time.sleep(0.1)
            
        except Exception as e:
            print(f"Fatal tracking error: {e}")
            self.error_state = True
        finally:
            # Always clean up resources
            self.cleanup()
            print("Tracking thread terminated")

    def return_frames(self):
        """Thread-safe method to get the latest labeled frames."""
        with self.tracking_lock:
            return self.labeled_frames
    
    def get_error_state(self):
        """Check if tracker encountered any fatal errors."""
        return self.error_state
    
    def stop(self):
        """Stop the tracking process safely."""
        print("Stopping tracking...")
        self.is_running = False
        # No need to call cleanup here, it's called in the finally block
    
    def cleanup(self):
        """Ensure the detection thread stops when done."""
        if self.detector.running:
            try:
                self.detector.stop()
                print("Detection thread stopped successfully.")
            except Exception as e:
                print(f"Error stopping detection thread: {e}")
