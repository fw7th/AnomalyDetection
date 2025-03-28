import cv2 as cv
import threading
import time
import numpy as np
import json
from typing import List
import supervision as sv
from src.prediction.detection import ObjectDetector as OD
from src.prediction.tracking import ObjectTracker as TK
import queue

CORNER_ANNOTATOR = sv.BoxCornerAnnotator()

class VideoProcessor:
    def __init__(self, source_path, save_dir: str):
        self.source_path = source_path
        self.save_dir = save_dir
        self.detect = OD(self.source_path)
        self.tracker = TK(self.source_path)
        
        # Use a thread-safe queue for frame buffering
        self.frame_queue = queue.Queue(maxsize=100)
        
        # Event to signal thread termination
        self.stop_event = threading.Event()

    def display_video(self):
        """Optimized video display with improved thread safety and performance."""
        # Create window with better performance settings
        cv.namedWindow("Tracking", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow("Tracking", 720, 480)
        
        # Start tracking in a separate thread
        tracking_thread = threading.Thread(target=self._tracking_worker)
        tracking_thread.daemon = True
        tracking_thread.start()
        
        try:
            no_frame_count = 0
            while not self.stop_event.is_set():
                try:
                    # Use a non-blocking get with a timeout
                    labeled_frame = self.frame_queue.get(timeout=1)
                    
                    # Ensure frame is valid and display
                    if labeled_frame is not None and labeled_frame.size > 0:
                        cv.imshow("Tracking", labeled_frame)
                        no_frame_count = 0
                    else:
                        no_frame_count += 1
                    
                    # Check for quit key with minimal blocking
                    key = cv.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    # Exit if no frames for extended period
                    if no_frame_count > 300:
                        print("No frames received. Exiting.")
                        break
                
                except queue.Empty:
                    print("Waiting for frames...")
                    continue
        
        finally:
            # Ensure clean shutdown
            self.stop_event.set()
            self.tracker.stop()
            cv.destroyAllWindows()

    def _tracking_worker(self):
        """Dedicated worker for tracking and frame queueing."""
        try:
            # Initialize tracker
            tracking_thread = threading.Thread(
                target=self.tracker.get_tracked_objects, 
            )
            tracking_thread.daemon = True
            tracking_thread.start()
            
            # Continuously retrieve and queue frames
            while not self.stop_event.is_set():
                labeled_frame = self.tracker.return_frames()
                
                if labeled_frame is not None:
                    # Non-blocking put with timeout to prevent queue blocking
                    try:
                        self.frame_queue.put(labeled_frame.copy(), timeout=0.1)
                    except queue.Full:
                        # If queue is full, remove oldest frame
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                
                # Prevent tight loop
                time.sleep(0.01)
        
        except Exception as e:
            print(f"Tracking worker error: {e}")
        finally:
            self.stop_event.set()

    def save_video(self):
        """Optimized video saving method."""
        # Collect frames from queue
        frames = []
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
                if frame is not None and frame.size > 0:
                    frames.append(frame)
            except queue.Empty:
                break
        
        if not frames:
            print("No frames to save.")
            return
        
        # Use VideoInfo for resolution and FPS
        try:
            vid_info = sv.VideoInfo.from_video_path(self.source_path)
            
            # Ensure all frames are same size
            size = (frames[0].shape[1], frames[0].shape[0])
            
            # Create video writer
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(
                self.save_dir, 
                fourcc, 
                vid_info.fps, 
                size
            )
            
            # Write frames
            for frame in frames:
                # Resize if necessary
                if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
                    frame = cv.resize(frame, size)
                out.write(frame)
            
            out.release()
            print(f"Video saved successfully to {self.save_dir}")
        
        except Exception as e:
            print(f"Video saving error: {e}")

            
def load_zones_config(file_path: str) -> List[np.ndarray]:
    """
    Load polygon zone configurations from a JSON file.

    This function reads a JSON file which contains polygon coordinates, and
    converts them into a list of NumPy arrays. Each polygon is represented as
    a NumPy array of coordinates.

    Args:
        file_path (str): The path to the JSON configuration file.

    Returns:
        List[np.ndarray]: A list of polygons, each represented as a NumPy array.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        return [np.array(polygon, np.int32) for polygon in data]
