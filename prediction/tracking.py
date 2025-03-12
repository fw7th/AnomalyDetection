import supervision as sv
from prediction.detection import ObjectDetector

LABEL_ANNOTATOR = sv.LabelAnnotator()

class Tracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.detector = ObjectDetector()

    def get_tracked_objects(self, source):
        if source is None:
            print("No video source provided.")

        
        cap, gen = self.detector.detect_objects(source)
        
        if gen and cap is not None:
            for detections, annotated, results in gen:
                if detections and results is None:
                    print("No detections found, check Video Stream.")
                    continue

                tracked_detections = self.tracker.update_with_detections(detections)

                labels = [
                    f"#{tracker_id} {results.names[class_id]}"
                    for class_id, tracker_id 
                    in zip(tracked_detections.class_id, tracked_detections.tracker_id)
                ]

                labeled_frames = LABEL_ANNOTATOR.annotate(annotated, detections=tracked_detections, labels=labels)
                yield labeled_frames
