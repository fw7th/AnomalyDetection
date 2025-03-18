from ultralytics import YOLO
import threading
import supervision as sv
from src import config
import cv2 as cv

CLASS = 0
VID_STRIDE = 1

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)
        self.frame = None
        self.detections = None
        self.annotated = None
        self.results = None
        self.running = False
        self.lock = threading.Lock() # Thread safety

    def detect_objects(self, source=None):
        if source is None:
            print("Error: No video source provided.")
            return None, None

        self.cap = cv.VideoCapture(source)
        if not self.cap.isOpened():
            print("Error: Failed to open video source.")

        
        self.running = True
        self.thread = threading.Thread(target=self._object_generator, daemon=True)
        self.thread.start()

        return self.cap, self  # Return cap & generator

    def _object_generator(self):
        from src.utils.general import BOX_ANNOTATOR
        """Runs detection in a separate thread while updating detections."""
        while self.running:
            success, frame = self.cap.read()
            if not success:
                break

            results = self.model(
                frame,
                classes=CLASS,
                vid_stride=VID_STRIDE
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            annotated = BOX_ANNOTATOR.annotate(frame.copy(), detections=detections)

            # Store the latest results with thread safety
            with self.lock:
                self.frame = frame
                self.detections = detections
                self.annotated = annotated
                self.results = results

        self.cap.release()
        cv.destroyAllWindows()

    def get_latest_results(self):
        """Safely get the latest detection results."""
        with self.lock:
            return self.detections, self.annotated, self.results

    def stop(self):
        """Stop the detection loop."""
        self.running = False
        self.thread.join()

