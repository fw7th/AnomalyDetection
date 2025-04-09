import numpy as np
import torch.cuda
import cv2 as cv
import threading
import queue
import time

class VideoDisplay:
    """
    Optimized video processing class that handles frame capture, processing,
    and display with efficient multithreading and resource management.
    """
    def __init__(self, tracker_queue, maxframes=1000, enable_saving=False, save_dir=None):
        """
        Initialize the video processor with source and output paths.
        
        Parameters:
        -----------
        source_path : str
            Path to the source video or camera index
        maxsize : int, optional
            Size of the frame buffer queue (default: 15)
        """
        self.tracker_queue = tracker_queue
        self.use_gpu = torch.cuda.is_available()
        
        # Thread-safe queue for frame buffering
        self.worker = None
        self.threads = []
        self.display_lock = threading.Lock()
        self.enable_saving = enable_saving
        self.save_dir = save_dir
        
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

    def display_video(self, window_name="Tracking"):
        """
        Display the processed video with optimized performance and provide
        options for saving frames.
        
        Parameters:
        -----------
        window_name : str, optional
            Name of the display window (default: "Tracking")
        """
        while self.running.is_set():
            try:
                frames = self.tracker_queue.get(timeout=0.1)
                if frames is None:
                    print("No tracking frames recieved")
                    time.sleep(1)

                with self.display.lock:
                    cv.imshow(window_name, frames)

                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            except:
                print("No frames to extract from tracker queue")
                time.sleep(1)

    def save(self):
        if self.enable_saving == True:
            pass

    def run(self):
        try:
            self.running.set()
            self.worker = threading.Thread(
                target=self.display_video,
                daemon=True
            )
            self.worker.start

        except KeyboardInterrupt:
            print("Processing interrupted by user")

    def stop(self):
        self.running.clear()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join()

        cv.destroyAllWindows()
