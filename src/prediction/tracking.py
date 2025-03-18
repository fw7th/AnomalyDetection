import supervision as sv
from src.prediction.detection import ObjectDetector
import time
from src.utils.time import ClockBasedTimer

LABEL_ANNOTATOR = sv.LabelAnnotator()

class Tracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.detector = ObjectDetector()
        self.timer = ClockBasedTimer()
        self.labeled_frames = None


    def get_tracked_objects(self, source=None):
        if source is None:
            print("No video source provided.")
            return
        
        cap, detector = self.detector.detect_objects(source)

        if cap is None:
            print("Error: Failed to initialize detection.")
            return

        while cap.isOpened():
            time.sleep(0.3)
            detections, annotated, results = detector.get_latest_results()

            if detections is None or results is None:
                print("No detections found yet...")
                continue

            tracked_detections = self.tracker.update_with_detections(detections)
            time_in_area = self.timer.tick(tracked_detections) # Calculate the time a human is present

            if len(tracked_detections.tracker_id) > 0 and annotated is not None:
                labels = [
                    f"#{tracker_id} {times/10}s"
                    for tracker_id, times in zip(tracked_detections.tracker_id, time_in_area)
                ]


                self.labeled_frames = LABEL_ANNOTATOR.annotate(annotated, detections=tracked_detections, labels=labels)

            if detector.get_latest_results() is None:
                break

        self.cleanup() # Clean up threading after video fully ends


    def return_frames(self):
        return self.labeled_frames


    def cleanup(self):
        """Ensure the detection thread stops when done."""
        if self.detector.running: # Only stop if thread is still running for no reason
            self.detector.stop() 
            print("Detection thread stopped successfully.")
