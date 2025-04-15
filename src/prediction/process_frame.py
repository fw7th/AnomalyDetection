from src.utils.preprocessing import preprocess_frame
import multiprocessing as mp
import threading
import torch
import time
import queue

class FrameProcessor_:
    def __init__(self, frame_queue, preprocessed_queue):
        self.use_gpu = torch.cuda.is_available()
        self.frame_queue = frame_queue
        self.preprocessed_queue = preprocessed_queue
        if self.use_gpu:
            self._running = threading.Event()
        else:
            self._running = mp.Event()
        
    def process_single_frame_cpu(self):
        """Process a single frame from the queue (CPU version)"""
        try:
            # Get a frame without blocking indefinitely
            frame = self.frame_queue.get(timeout=0.5)
            print("Frame received for preprocessing [CPU].")
            
            # Process the frame
            preprocessed_frame = preprocess_frame(frame)
            print("Preprocessing Complete [CPU]")
            
            # Put the processed frame in the output queue
            if preprocessed_frame.size > 0:
                self.preprocessed_queue.put(preprocessed_frame, timeout=0.5)
                print("Frame sent to detect_pipe") 
            else:
                print("Preprocessed frame invalid [CPU].")
                
            return True  # Successfully processed a frame
            
        except queue.Empty:
            # No frames available, just return
            print("No frames recieved from frame reader")
            return False

        except queue.Full:
            print("Object Detection is slow or has an issue")

        except Exception as e:
            print(f"[CPU] Preprocessing error: {e}")
            time.sleep(0.1)
            return False
        
    def process_single_frame_gpu(self):
        """Process a single frame from the queue (GPU version)"""
        try:
            # Get a frame without blocking indefinitely
            frame = self.frame_queue.get(timeout=0.1)
            print("Frame received for preprocessing [GPU].")
            
            # Process the frame
            preprocessed_frame = preprocess_frame(frame)
            print("Preprocessing Complete [GPU]")
            
            # Put the processed frame in the output queue
            if preprocessed_frame.size > 0:
                self.preprocessed_queue.put(preprocessed_frame, timeout=0.5)
                print("Frame sent to detect_pipe")
            else:
                print("Preprocessed frame invalid [GPU].")
                
            return True  # Successfully processed a frame
            
        except queue.Empty:
            # No frames available, just return
            return False
        except Exception as e:
            print(f"[GPU] Preprocessing error: {e}")
            time.sleep(0.1)
            return False
            
    # Add these methods for backward compatibility with your existing code
    def preprocess_cpu(self):
        """Legacy method for CPU preprocessing loop"""
        while self._running.is_set():
            self.process_single_frame_cpu()
            
    def preprocess_gpu(self):
        """Legacy method for GPU preprocessing loop"""
        while self._running.is_set():
            self.process_single_frame_gpu()
