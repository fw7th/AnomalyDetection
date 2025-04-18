import cv2 as cv
import threading
import time
import queue

class Frames:
    def __init__(self, frame_queue, source):
        self.source = source
        self.lock = threading.Lock()
        self.frame_queue = frame_queue
        self.running = threading.Event()
        self.cap = None
        self.fps_target = None  # Will be set based on source fps
        self.frame_delay = 0  # Time to wait between frame reads
        self.skip_stride = 2
        
    def read_frames(self): 
        if self.source is None:
            print("Error: No video source provided.")
            return 
       
        try:
            self.cap = cv.VideoCapture(self.source)
            if not self.cap.isOpened():
                print("Error: Failed to open video source.")
                return
            
            # Get source frame rate and calculate frame delay
            source_fps = self.cap.get(cv.CAP_PROP_FPS)
            if source_fps > 0:
                self.frame_delay = 1.0 / source_fps
                print(f"Source FPS: {source_fps}, Frame delay: {self.frame_delay:.4f}s")
            else:
                # Default for live streams or when FPS is unavailable
                self.frame_delay = 0.001
                print("Using default frame delay for live stream")
            
            frame_count = 0
            last_log_time = time.time()
            
            while self.running.is_set():
                try:
                    # Read frame with timeout
                    start_time = time.time()
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        print("End of video stream reached or frame read error")
                        break
                    
                    # Log FPS periodically
                    frame_count += 1
                    current_time = time.time()
                    if current_time - last_log_time >= 2.0:  # Log every 5 seconds
                        elapsed = current_time - last_log_time
                        fps = frame_count / elapsed
                        print(f"Reading at {fps:.2f} FPS")
                        frame_count = 0
                        last_log_time = current_time
                    
                    # Put frame in queue with timeout to prevent blocking indefinitely
                    try:
                        if frame_count % self.skip_stride == 0:
                            self.frame_queue.put(frame, timeout=0.1)

                    except queue.Full:
                        print("Frame queue full, preprocessing slow, skipping frame")
                        continue
                    
                    # Control frame rate (respect source FPS or skip if processing is too slow)
                    processing_time = time.time() - start_time
                    if processing_time < self.frame_delay:
                        time.sleep(self.frame_delay - processing_time)
                    
                except Exception as e:
                    print(f"Error reading frame: {e}")
                    # Short delay to prevent tight loop on errors
                    time.sleep(0.1)
                    # Try to recover rather than breaking out
                    continue
                    
        except Exception as e:
            print(f"Critical error in frame reader: {e}")
