from src.utils.preprocessing import preprocess_frame
import multiprocessing as mp
# import threading
import torch
import time
import queue

# Set PyTorch thread count to optimize for 2 cores
torch.set_num_threads(2)

class FrameProcessor_:
    def __init__(self, frame_queue, preprocessed_queue, batch_size=3):
        self.use_gpu = torch.cuda.is_available()
        self.frame_queue = frame_queue
        self.preprocessed_queue = preprocessed_queue
        self.batch_size = batch_size
        self._running = mp.Event()
        self.frame_count = 0
        self.last_log_time = time.time()
        self.mutex = mp.Manager().RLock()
        
    def process_frame(self):
        """Process a batch of frames from the queue"""
        batch_frames = []
        
        # Try to fill the batch
        start_time = time.time()
        while len(batch_frames) < self.batch_size:
            try:
                self.mutex.acquire()
                # Set a short timeout to avoid waiting too long for frames
                frame = self.frame_queue.get(timeout=0.01)
                batch_frames.append(frame)
                self.mutex.release()

            except queue.Empty:
                # If queue is empty, process whatever we've collected so far
                if len(batch_frames) > 0:
                    break
                else:
                    # No frames available at all
                    time.sleep(0.01)
                    return False
            
            # Avoid waiting too long for a complete batch
            if time.time() - start_time > 0.2 and len(batch_frames) > 0:
                break
        
        try:
            # Process the batch of frames
            if len(batch_frames) > 0:
                preprocessed_frames = []
                
                for frame in batch_frames:
                    # Process each frame
                    preprocessed = preprocess_frame(frame)
                    if preprocessed.size > 0:
                        preprocessed_frames.append(preprocessed)
                
                # Update frame count for FPS calculation
                self.frame_count += len(preprocessed_frames)
                
                # Log FPS periodically
                current_time = time.time()
                if current_time - self.last_log_time >= 2.0:
                    elapsed = current_time - self.last_log_time
                    fps = self.frame_count / elapsed
                    print(f"Preprocessing at {fps:.2f} FPS (batch size: {len(batch_frames)})")
                    self.frame_count = 0
                    self.last_log_time = current_time
                
                # Put the processed frames in the output queue
                self.mutex.acquire()
                for frame in preprocessed_frames:
                    # Use a short timeout to avoid blocking if detection is slow
                    self.preprocessed_queue.put(frame, timeout=0.01)
                self.mutex.release()
                
                return True
            
            return False
            
        except queue.Full:
            print("Detection queue is full - preprocessing outpacing detection")
            time.sleep(0.01)
            return False
        except Exception as e:
            print(f"Preprocessing error: {e}")
            time.sleep(0.01)
            return False
    
    # Legacy method for compatibility
    def preprocess_cpu(self):
        self.preprocess_loop()
