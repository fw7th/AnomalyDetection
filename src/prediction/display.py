import cv2 as cv
import threading
import time
import numpy as np
import queue

class VideoDisplay:
    """
    Optimized video processing class that handles frame capture, processing,
    and display with efficient multithreading and resource management.
    """
    def __init__(self, detection_queue, maxframes=1000, enable_saving=False, save_dir=None):
        """
        Initialize the video processor with source and output paths.
        
        Parameters:
        -----------
        source_path : str
            Path to the source video or camera index
        maxsize : int, optional
            Size of the frame buffer queue (default: 15)
        """
        self.detection_queue = detection_queue
        
        # Thread-safe queue for frame buffering
        self.display_lock = threading.Lock()
        self.enable_saving = enable_saving
        self.save_dir = save_dir
        self.last_log_time = time.time()
        self.frame_count = 0
        
        # Event to signal thread termination
        self.running = threading.Event()
        self.is_saving = threading.Event()
        self.save_frames_lock = threading.Lock()
       
        # Store frames for saving (optional)
        self.save_frames = []
        self.max_save_frames = maxframes  # Limit to prevent memory issues
        self.save_thread = None

        if self.enable_saving and not self.save_dir:
            raise ValueError("You must specify a save_dir when enable_saving is True.")

    def display_video(self, window_name="detection"):
        """
        Display the processed video with optimized performance and provide
        options for saving frames.
        
        Parameters:
        -----------
        window_name : str, optional
            Name of the display window (default: "detection")
        """
        try:
            frame = self.detection_queue.get_nowait()
            assert isinstance(frame, np.ndarray), f"Frame isn't valid, {type(frame)}"

            if frame is None and frame.size == 0:
                print("No detection frames recieved, or frame is invalid")
                time.sleep(0.001)
                
            self.frame_count += 1 
            current_time = time.time()
            if current_time - self.last_log_time >= 2.0:
                elapsed = current_time - self.last_log_time
                fps = self.frame_count / elapsed
                print(f"Displaying at {fps:.2f} FPS ")
                self.frame_count = 0
                self.last_log_time = current_time

            cv.imshow(window_name, frame)
        
        except queue.Empty:
            print("Detector module is slow, add more power")

        except Exception as e:
            print(f"No frames to extract from detection queue: {e}")
            time.sleep(1)

    def save(self):
        if self.enable_saving == True:
            pass

    def should_exit(self):
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            return True
