import supervision as sv
import cv2 as cv
from inference.detection import ObjectDetector as OD
from inference.tracking import Tracker as TK

BOX_ANNOTATOR = sv.BoxAnnotator(
    thickness=2,
)

class VideoProcessor:
    def __init__(self, source_path : str, save_dir : str):
        self.source_path = source_path
        self.save_dir = save_dir
        self.detect = OD()
        self.tracker = TK()

    def process_detect(self):
        cap, gen = self.detect.detect_objects(self.source_path)
        if cap and gen is not None:
            vid_width = int(cap.get(3))
            vid_height = int(cap.get(4))
            vid_fps = cap.get(5)

            size = (vid_width, vid_height)

            result = cv.VideoWriter(
                self.save_dir,
                cv.VideoWriter_fourcc(*"mp4v"),
                vid_fps,
                size
            )
            for detections, annotated, results in gen:
                if annotated is None:
                    print("No detections found, check video stream.")
                    continue

                result.write(annotated)

            print("The video was successfully saved") 
            result.release()

    def process_track(self):
        labeled_frames = self.tracker.get_tracked_objects(self.source_path)

        vid_info = sv.VideoInfo.from_video_path(self.source_path)
        width, height = vid_info.resolution_wh
        size = (width, height)
        fps = vid_info.fps

        result2 = cv.VideoWriter(
            self.save_dir,
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            size
        )
        for frames in labeled_frames:
            if frames is None:
                print("No Tracked Objects.")
                continue

            result2.write(frames)

        print("The video was successfully saved") 
        result2.release()



