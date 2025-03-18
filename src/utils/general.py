import cv2 as cv
import supervision as sv
from src.prediction.detection import ObjectDetector as OD
from src.prediction.tracking import Tracker as TK

BOX_ANNOTATOR = sv.BoxAnnotator(
    thickness=2
)

class VideoProcessor:
    def __init__(self, source_path: str, save_dir: str):
        self.source_path = source_path
        self.save_dir = save_dir
        self.detect = OD()
        self.tracker = TK()
        self.frame_buffer = []  # Store frames for later saving

    def display_video(self):
        """Runs tracking, displays video in real-time, and stores frames for later saving."""
        self.tracker.get_tracked_objects(self.source_path)

        while True:
            labeled_frame = self.tracker.return_frames()
            if labeled_frame is None:
                print("No Tracked Objects.")
                break

            if labeled_frame is not None and labeled_frame.size > 0:
                cv.imshow("Tracking", labeled_frame)  # Show video
                self.frame_buffer.append(labeled_frame)  # Store frames
            else:
                print("Frame is empty or invalid.")

            if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

        cv.destroyAllWindows()

    def save_video(self):
        """Saves the processed video using the stored frames."""
        if not self.frame_buffer:
            print("No frames to save.")
            return
        
        vid_info = sv.VideoInfo.from_video_path(self.source_path)
        size = vid_info.resolution_wh
        fps = vid_info.fps

        result = cv.VideoWriter(
            self.save_dir,
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            size
        )

        for frame in self.frame_buffer:
            result.write(frame)

        result.release()
        print("The video was successfully saved.")

