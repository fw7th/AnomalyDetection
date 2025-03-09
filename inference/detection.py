from ultralytics import YOLO
import supervision as sv
import config

CLASS = 0
VID_STRIDE = 1

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)

    def detect_objects(self, source=None):
        import cv2 as cv
        if source is None:
            print("Error: No video source provided.")
            return None, None

        cap = cv.VideoCapture(source)

        if not cap.isOpened():
            print("Error: Failed to open video source.")
            return None, None

        return cap, self._object_generator(cap)  # Return cap & generator

    def _object_generator(self, cap):
        import cv2 as cv
        from utils import BOX_ANNOTATOR
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = self.model(
                frame,
                classes=CLASS,
                vid_stride=VID_STRIDE
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            annotated = BOX_ANNOTATOR.annotate(frame.copy(), detections=detections)
            

            yield detections, annotated, results 

        cap.release()
        cv.destroyAllWindows()
