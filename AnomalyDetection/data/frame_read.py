
"""
frame_read.py

Handles reading frames from a video source using OpenCV in a separate thread.
Useful for live or recorded video feeds, with built-in queueing and FPS control.
"""

from core import LOG
import cv2 as cv
import threading
import time
import queue

class Frames:
    """
    A threaded frame reader for video streams.

    Attributes:
        source (str or int): The video source (path or camera index).
        frame_queue (queue.Queue): Queue to store read frames.
        running (threading.Event): Flag to control the read loop.
        cap (cv2.VideoCapture): OpenCV video capture object.
        fps_target (float): Target frames per second (optional).
        frame_delay (float): Time to wait between frames based on source FPS.
        skip_stride (int): Number of frames to skip between reads.
        break_time (float): Time since source was lost.
    """
    
    def __init__(self, frame_queue, source):
        """
        Initializes the frame reader.

        Args:
            frame_queue (queue.Queue): A queue to hold frames for processing.
            source (str or int): Path to video file or camera index.
        """
        self.source = source
        self.frame_queue = frame_queue
        self.running = threading.Event()
        self.cap = None
        self.fps_target = None
        self.frame_delay = 0
        self.skip_stride = 2
        self.break_time = 0

    def read_frames(self):
        """
        Starts reading frames from the video source.

        Reads frames in a loop until stopped, skips frames based on stride,
        and pushes valid frames to the queue. Attempts to respect source FPS.
        """
        if self.source is None:
            LOG.critical("Error: No video source provided.")
            return

        try:
            self.cap = cv.VideoCapture(self.source)
            if not self.cap.isOpened():
                LOG.critical("Error: Failed to open video source.")
                return

            source_fps = self.cap.get(cv.CAP_PROP_FPS)
            self.frame_delay = 1.0 / source_fps if source_fps > 0 else 0.001
            LOG.info(f"Source FPS: {source_fps or 'Unknown'}, Frame delay: {self.frame_delay:.4f}s")

            frame_count = 0
            last_log_time = time.time()

            self.running.set()

            while self.running.is_set():
                start_time = time.time()
                ret, frame = self.cap.read()

                if not ret:
                    # If source is lost, start break timer
                    if self.break_time == 0:
                        self.break_time = time.time()
                    elif time.time() - self.break_time >= 7:
                        break
                    continue
                else:
                    self.break_time = 0

                frame_count += 1
                current_time = time.time()
                if current_time - last_log_time >= 5.0:
                    fps = frame_count / (current_time - last_log_time)
                    LOG.info(f"Reading at {fps:.2f} FPS")
                    frame_count = 0
                    last_log_time = current_time

                try:
                    if frame_count % self.skip_stride == 0:
                        self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    time.sleep(0.001)

                processing_time = time.time() - start_time
                if processing_time < self.frame_delay:
                    time.sleep(self.frame_delay - processing_time)

        except Exception as e:
            LOG.critical(f"Critical error in frame reader: {e}")
