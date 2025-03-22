import cv2 as cv
import threading
import time
import numpy as np
import supervision as sv
from src.prediction.detection import ObjectDetector as OD
from src.prediction.tracking import Tracker as TK

BOX_ANNOTATOR = sv.BoxAnnotator(
    thickness=2
)

class VideoProcessor:
    def __init__(self, source_path, save_dir: str):
        self.source_path = source_path
        self.save_dir = save_dir
        self.detect = OD()
        self.tracker = TK()
        self.frame_buffer = []  # Store frames for later saving


    def display_video(self):
        """Runs tracking, displays video in real-time, and stores frames for later saving."""
        # Create window explicitly before starting tracking
        cv.namedWindow("Tracking", cv.WINDOW_NORMAL)
        cv.moveWindow("Tracking", 100, 100)
        
        # Force initial window update
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv.putText(blank_frame, "Waiting for video...", (50, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.imshow("Tracking", blank_frame)
        cv.waitKey(100)  # Give time for window to appear
        
        # Start tracking in a separate thread
        tracking_thread = threading.Thread(target=self.tracker.get_tracked_objects, args=(self.source_path,))
        tracking_thread.daemon = True
        tracking_thread.start()
        
        print("Starting display loop...")
        
        no_frame_count = 0
        while True:
            labeled_frame = self.tracker.return_frames()
            if labeled_frame is None:
                no_frame_count += 1
                print(f"Waiting for frames... {no_frame_count}/30")
                if no_frame_count > 30:
                    print("No frames received after multiple attempts. Exiting.")
                    break
                cv.waitKey(100)  # Longer wait to allow events to process
                continue
            
            # Reset the counter when we get a frame
            no_frame_count = 0
            
            # Ensure frame is in correct format before display
            if labeled_frame.size > 0:
                labeled_frame = labeled_frame.astype(np.uint8)
                cv.imshow("Tracking", labeled_frame)
                print("Displayed frame")
                
                # Save a test frame to verify it's valid
                if len(self.frame_buffer) == 0:
                    cv.imwrite("test_frame.jpg", labeled_frame)
                    print("Saved test frame")
                    
                self.frame_buffer.append(labeled_frame.copy())
            else:
                print("Frame is empty or invalid")
            
            # Use a longer wait key to ensure event processing
            key = cv.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Quit key pressed!")
                break
        
        print("Display loop ended")
        self.tracker.stop()
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
