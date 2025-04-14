from src.utils.preprocessing import preprocess_frame
import multiprocessing as mp
import threading
import torch
import time

class FrameProcessor_:
    def __init__(self, frame_queue, preprocessed_queue):
        self.use_gpu = torch.cuda.is_available()
        self.frame_queue = frame_queue
        self.preprocessed_queue = preprocessed_queue

        if self.use_gpu:
            self._running = threading.Event()
        else:
            self._running = mp.Event()

    def preprocess_cpu(self):
        try:
            while self._running.is_set():
                frame = self.frame_queue.get(timeout=0.1)
                print("Frame received for preprocessing [CPU].")

                preprocessed_frame = preprocess_frame(frame)
                print("Preprocessing Complete [CPU]")

                if preprocessed_frame.size > 0:
                    self.preprocessed_queue.put(preprocessed_frame)
                    print("Frame sent to detect_pipe") 
                else:
                    print("Preprocessed frame invalid [CPU].")

        except Exception as e:
            print(f"[CPU] Preprocessing error: {e}")
            time.sleep(0.2)
        
    def preprocess_gpu(self):
        while self._running.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5)
                print("Frame received for preprocessing [GPU].")

                preprocessed_frame = preprocess_frame(frame)
                print("Preprocessing Complete [GPU]")

                if preprocessed_frame.size > 0:
                    self.preprocessed_queue.put(preprocessed_frame)
                else:
                    print("Preprocessed frame invalid [GPU].")

            except Exception as e:
                print(f"[GPU] Preprocessing error: {e}")
                time.sleep(0.2)
