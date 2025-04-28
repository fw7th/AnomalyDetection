"""
display.py

Module contains multithreaded implementations for displaying frames,
and saving the source frames as a video while ensuring thread safety.

Author: fw7th
Date: 2025-04-26
"""

from core import LOG
import cv2 as cv
import threading
import time
import numpy as np
import queue

class VideoDisplay:
    """
    Optimized video processing class that handles frame capture, processing,
    and display with efficient multithreading and resource management.

    Attributes:
    -----------
    detection_queue : queue.Queue
        Thread-safe queue for passing frames to be processed and displayed.
    enable_saving : bool, optional
        Flag to enable saving of processed frames (default: False).
    save_dir : str, optional
        Directory where video frames will be saved (requires enable_saving=True).
    last_log_time : float
        Timestamp for the last log.
    frame_count : int
        Counter for the number of frames displayed.
    running : threading.Event
        Event to signal when the video processing is running.
    saving_started : threading.Event
        Event to signal when the saving thread has started.
    save_buffer : queue.Queue
        Buffer to store frames for saving to disk.
    save_thread : threading.Thread, optional
        Thread to handle saving frames to disk.
    frame_size : tuple, optional
        The size (width, height) of the frames to be saved.
    last_frame_save : float
        Timestamp for the last frame saved to disk.
    """

    def __init__(self, detection_queue, enable_saving=False, save_dir=None):
        """
        Initializes the video processor with source and output paths.

        Args:
        -----
        detection_queue : queue.Queue
            Thread-safe queue for passing frames to be processed and displayed.
        enable_saving : bool, optional
            Flag to enable saving of processed frames (default: False).
        save_dir : str, optional
            Directory where video frames will be saved (requires enable_saving=True).

        Raises:
        -------
        ValueError
            If enable_saving is True but save_dir is not provided.
        """
        self.detection_queue = detection_queue
        self.enable_saving = enable_saving
        self.save_dir = save_dir
        self.last_log_time = time.time()
        self.frame_count = 0
        self.running = threading.Event()
        self.saving_started = threading.Event()
        self.save_buffer = queue.Queue(maxsize=20)
        self.save_thread = None
        self.frame_size = None
        self.last_frame_save = time.time()

        if self.enable_saving and not self.save_dir:
            raise ValueError("You must specify a save_dir when enable_saving is True.")

    def display_video(self, window_name="detection"):
        """
        Displays the processed video with optimized performance and provides
        options for saving frames.

        Args:
        -----
        window_name : str, optional
            Name of the display window (default: "detection").

        Notes:
        ------
        If saving is enabled, frames are buffered for saving. The FPS (frames per second)
        is logged every 2 seconds.
        """
        cv.namedWindow(window_name, cv.WINDOW_FULLSCREEN)
        try:
            self.frame = self.detection_queue.get(timeout=0.01)
            assert isinstance(self.frame, np.ndarray), f"Frame isn't valid, {type(self.frame)}"
            
            if self.enable_saving and self.frame.size > 0:
                try:
                    self.save_buffer.put_nowait(self.frame.copy())
                    self.frame_size = (self.frame.shape[1], self.frame.shape[0])
                except queue.Full:
                    pass

            if self.frame is None or self.frame.size == 0:
                LOG.warning("No detection frames received, or frame is invalid")
                time.sleep(0.001)
                return

            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_log_time >= 5.0:
                elapsed = current_time - self.last_log_time
                fps = self.frame_count / elapsed
                LOG.info(f"Displaying at {fps:.2f} FPS ")
                self.frame_count = 0
                self.last_log_time = current_time

            cv.imshow(window_name, self.frame)

            if self.enable_saving and not self.saving_started.is_set():
                self.start_saving_thread()
                self.saving_started.set()

        except queue.Empty:
            time.sleep(0.001)

        except Exception as e:
            LOG.error(f"Error: Issue with Display {e}")
            time.sleep(0.01)

    def save(self):
        """
        Thread function that saves frames to a video file.

        This method is run in a separate thread to handle saving frames asynchronously.
        It creates a VideoWriter and writes buffered frames to the specified file.
        """
        fps = 15
        result = None
        
        LOG.info("Save thread started")

        wait_start = time.time()
        while self.frame_size is None:
            if not self.running.is_set() or time.time() - wait_start > 10:
                LOG.error("Timed out waiting for frame size or thread stopped")
                return
            time.sleep(0.01)
        
        try:
            LOG.info(f"Creating video writer with size {self.frame_size}")
            result = cv.VideoWriter(
                self.save_dir,
                cv.VideoWriter_fourcc(*"mp4v"),
                fps,
                self.frame_size
            )
            
            if not result.isOpened():
                LOG.error(f"Failed to open video writer. Check codec and path: {self.save_dir}")
                return
                
            LOG.info("VideoWriter created successfully")
            
            frame_count = 0
            while self.running.is_set():
                try:
                    frame = self.save_buffer.get(timeout=0.01)
                    result.write(frame)
                    frame_count += 1
                    self.last_frame_save = time.time()
                    
                    if frame_count % 100 == 0:
                        LOG.debug(f"Saved {frame_count} frames to video")
                        
                except queue.Empty:
                    if time.time() - self.last_frame_save > 5 and not self.running.is_set():
                        LOG.info("No new frames to save for 5 seconds and not running, exiting save thread")
                        break
                except Exception as e:
                    LOG.warning(f"Error writing frame to video: {e}")
                    time.sleep(0.01)
            
            LOG.info(f"Save thread completing, saved {frame_count} frames")
                
        except Exception as e:
            LOG.warning(f"Error in save thread: {e}")
        finally:
            if result is not None:
                result.release()
                LOG.info(f"Video saved to {self.save_dir}")
            self.saving_started.clear()

    def start_saving_thread(self):
        """
        Starts the saving thread if it's not already running.

        This method initiates the save thread which handles the saving of frames
        to disk asynchronously. It only starts the thread if it isn't already running.
        """
        if self.save_thread and self.save_thread.is_alive():
            LOG.debug("Save thread already running")
            return
        
        LOG.debug("Starting thread to save inference video")
        self.save_thread = threading.Thread(target=self.save, daemon=True)
        self.save_thread.start()

    def should_exit(self):
        """
        Checks if the user wants to exit (pressed 'q').

        Returns:
        --------
        bool
            True if 'q' was pressed, False otherwise.
        """
        key = cv.waitKey(1) & 0xFF
        return key == ord('q')

    def cleanup(self):
        """
        Cleans up resources when shutting down.

        This method clears the running event and waits for the save thread to
        complete before closing any windows.
        """
        self.running.clear()
        if self.save_thread and self.save_thread.is_alive():
            LOG.debug("Waiting for save thread to complete...")
            self.save_thread.join(timeout=5.0)
        cv.destroyAllWindows()
