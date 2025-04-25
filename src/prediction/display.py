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
    def __init__(self, detection_queue, enable_saving=False, save_dir=None):
        """
        Initialize the video processor with source and output paths.
        
        Parameters:
        -----------
        """
        self.detection_queue = detection_queue
        
        # Thread-safe queue for frame buffering
        self.enable_saving = enable_saving
        self.save_dir = save_dir
        self.last_log_time = time.time()
        self.frame_count = 0
        
        # Events to signal thread termination and saving status
        self.running = threading.Event()
        self.saving_started = threading.Event()
       
        # Store frames for saving (optional)
        self.save_buffer = queue.Queue(maxsize=20)
        self.save_thread = None
        self.frame_size = None
        self.last_frame_save = time.time()

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
            self.frame = self.detection_queue.get(timeout=0.01)
            assert isinstance(self.frame, np.ndarray), f"Frame isn't valid, {type(self.frame)}"
           
            if self.enable_saving and self.frame.size > 0:
                try:
                    self.save_buffer.put_nowait(self.frame.copy())
                    self.frame_size = (self.frame.shape[1], self.frame.shape[0])
                except queue.Full:
                    # If buffer is full, just continue without saving this frame
                    pass

            if self.frame is None or self.frame.size == 0:
                print("No detection frames received, or frame is invalid")
                time.sleep(0.001)
                return
                
            self.frame_count += 1 
            current_time = time.time()
            if current_time - self.last_log_time >= 2.0:
                elapsed = current_time - self.last_log_time
                fps = self.frame_count / elapsed
                print(f"Displaying at {fps:.2f} FPS ")
                self.frame_count = 0
                self.last_log_time = current_time

            cv.imshow(window_name, self.frame)

            # Start the saving thread only once
            if self.enable_saving and not self.saving_started.is_set():
                self.start_saving_thread()
                self.saving_started.set()

        except queue.Empty:
            # No frames available right now
            time.sleep(0.001)

        except Exception as e:
            print(f"Error: Issue with Display {e}")
            time.sleep(0.01)

    def save(self):
        """Thread function that saves frames to video file"""
        fps = 15
        result = None
        
        print("Save thread started")
        
        # Wait for frame_size to be set before creating VideoWriter
        wait_start = time.time()
        while self.frame_size is None:
            if not self.running.is_set() or time.time() - wait_start > 10:
                print("Timed out waiting for frame size or thread stopped")
                return
            time.sleep(0.01)
        
        try:
            print(f"Creating video writer with size {self.frame_size}")
            result = cv.VideoWriter(
                self.save_dir,
                cv.VideoWriter_fourcc(*"mp4v"),
                fps,
                self.frame_size
            )
            
            if not result.isOpened():
                print(f"Failed to open video writer. Check codec and path: {self.save_dir}")
                return
                
            print("VideoWriter created successfully")
            
            # Keep running while the main thread is running
            frame_count = 0
            while self.running.is_set():
                try:
                    frame = self.save_buffer.get(timeout=0.01)
                    result.write(frame)
                    frame_count += 1
                    self.last_frame_save = time.time()
                    
                    # Occasionally report progress
                    if frame_count % 100 == 0:
                        print(f"Saved {frame_count} frames to video")
                        
                except queue.Empty:
                    # No new frames to save right now
                    # Only exit if we haven't received frames for a while
                    if time.time() - self.last_frame_save > 5 and not self.running.is_set():
                        print("No new frames to save for 5 seconds and not running, exiting save thread")
                        break
                except Exception as e:
                    print(f"Error writing frame to video: {e}")
                    # Don't break on transient errors
                    time.sleep(0.01)
            
            print(f"Save thread completing, saved {frame_count} frames")
                
        except Exception as e:
            print(f"Error in save thread: {e}")
        finally:
            # Always clean up resources
            if result is not None:
                result.release()
                print(f"Video saved to {self.save_dir}")
            self.saving_started.clear()
        
    def start_saving_thread(self):
        """Start the saving thread if it's not already running"""
        if self.save_thread and self.save_thread.is_alive():
            print("Save thread already running")
            return
        
        print("Starting thread to save inference video")
        self.save_thread = threading.Thread(target=self.save, daemon=True)
        self.save_thread.start()
        
    def should_exit(self):
        """Check if user wants to exit (pressed 'q')"""
        key = cv.waitKey(1) & 0xFF
        return key == ord('q')
        
    def cleanup(self):
        """Clean up resources when shutting down"""
        self.running.clear()
        if self.save_thread and self.save_thread.is_alive():
            print("Waiting for save thread to complete...")
            # Give it a reasonable time to finish
            self.save_thread.join(timeout=5.0)
        cv.destroyAllWindows()
