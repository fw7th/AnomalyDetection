
"""
process_frame.py

This module handles batch preprocessing of video frames from a shared queue.
Frames are optionally processed on a GPU (if available) using PyTorch-compatible preprocessing,
and pushed into a second queue for further inference or downstream tasks.

Typical usage:
    processor = FrameProcessor_(frame_queue, preprocessed_queue)
    while True:
        processor.process_frame()
"""

from utils import preprocess_frame
from core import LOG
import multiprocessing as mp
import threading
import torch
import time
import queue

# Set PyTorch thread count to optimize for constrained CPUs
torch.set_num_threads(2)


class FrameProcessor_:
    """
    Handles frame preprocessing in batches using multiprocessing-safe queues.
    
    Attributes:
        frame_queue (mp.Queue): Source queue containing raw frames.
        preprocessed_queue (mp.Queue): Destination queue for preprocessed frames.
        batch_size (int): Number of frames to process per batch.
        use_gpu (bool): Flag indicating CUDA availability.
        frame_count (int): Tracks total frames processed (for logging).
        fps (float): Calculated processing FPS over intervals.
        mutex (mp.RLock): Multiprocessing-safe lock for queue access.
    """
    
    def __init__(self, frame_queue, preprocessed_queue, batch_size=3):
        self.use_gpu = torch.cuda.is_available()
        self.frame_queue = frame_queue
        self.preprocessed_queue = preprocessed_queue
        self.batch_size = batch_size
        self._running = mp.Event()
        self.frame_count = 0
        self.fps = 0
        self.last_log_time = time.time()
        self.mutex = mp.Manager().RLock()

    def process_frame(self):
        """
        Reads a batch of frames from the frame queue, applies preprocessing,
        and pushes results to the preprocessed queue. Also handles logging and
        frame rate control.

        Returns:
            bool: True if processing occurred, False if skipped due to empty queues or delay.
        """
        batch_frames = []
        start_time = time.time()

        # Fill the batch
        while len(batch_frames) < self.batch_size:
            try:
                self.mutex.acquire()
                frame = self.frame_queue.get(timeout=0.01)
                batch_frames.append(frame)
                self.mutex.release()
            except queue.Empty:
                if len(batch_frames) > 0:
                    continue 
                else:
                    time.sleep(0.01)
                    return False
            except EOFError or BrokenPipeError:
                LOG.info("End of stream reached")
                break
            if time.time() - start_time > 0.2 and len(batch_frames) > 0:
                break

        try:
            # Apply preprocessing
            if len(batch_frames) > 0:
                preprocessed_frames = [
                    preprocess_frame(frame)
                    for frame in batch_frames
                    if preprocess_frame(frame).size > 0
                ]

                if len(batch_frames) < 0 or frame.size < 0:
                        raise ValueError("Stream has ended.")

                self.frame_count += len(preprocessed_frames)

                # Periodic FPS logging
                current_time = time.time()
                elapsed = current_time - self.last_log_time
                self.fps = self.frame_count / elapsed
                if current_time - self.last_log_time >= 5.0:
                    LOG.info(f"Preprocessing at {self.fps:.2f} FPS (batch size: {len(batch_frames)})")
                    self.frame_count = 0
                    self.last_log_time = current_time

                # Push to output queue
                self.mutex.acquire()
                for frame in preprocessed_frames:
                    self.preprocessed_queue.put(frame, timeout=0.01)
                self.mutex.release()

            # Maintain source-aligned framerate
            self.frame_delay = 1.0 / self.fps
            processing_time = time.time() - start_time
            if processing_time < self.frame_delay:
                time.sleep(self.frame_delay - processing_time)
                return True

            return False

        except queue.Full:
            LOG.debug("Detection queue is full")
            time.sleep(0.001)
            return False
        except ValueError or EOFError or BrokenPipeError:
            LOG.info("Stream has ended, ending preprocessor.")
        except Exception as e:
            LOG.error(f"Preprocessing error: {e}")
            time.sleep(0.001)
            return False

    def preprocess_cpu(self):
        """
        Legacy stub method for backward compatibility with older codebases.
        Currently just calls `preprocess_loop()` if implemented.
        """
        self.preprocess_loop()
