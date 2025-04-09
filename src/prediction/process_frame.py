from src.utils.preprocessing import preprocess_frame
import queue as std_queue
import multiprocessing
import threading
import torch
import time

class FrameProcessor_:
    def __init__(self, frame_queue, maxsize=25):
        self.frame_queue = frame_queue
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.preprocessed_queue = std_queue.Queue(maxsize)
            self.thread_running = threading.Event()
            self.thread_worker = None
        else:
            self.preprocessed_queue = multiprocessing.Queue(maxsize)
            self.process_running = multiprocessing.Event()
            self.processes = []

    def start(self):
        if self.use_gpu:
            self.thread_running.set()
            self.thread_worker = threading.Thread(target=self.preprocess_gpu, daemon=True)
            self.thread_worker.start()
            print("Preprocessor thread initialized")
    
        else:
            self.process_running.set()
            for _ in range(2):
                p = self.process_worker = multiprocessing.Process(target=self.preprocess_cpu,)
                self.processes.append(p)

            for p in self.processes:
                p.start()
                self.process_running.wait()  # Wait until initialization complete
            print("Preprocessor initialized")

    def preprocess_cpu(self):
        while self.process_running.is_set():
            try:
                print(f"[{time.time()}] Frame reader starting...")
                frame = self.frame_queue.get(timeout=0.5)
                print("Frame received for preprocessing [CPU].")

                preprocessed_frame = preprocess_frame(frame)
                print("Preprocessing Complete [CPU]")

                if preprocessed_frame.size > 0:
                    self.preprocessed_queue.put(preprocessed_frame, timeout=0.5)
                else:
                    print("Preprocessed frame invalid [CPU].")

            except Exception as e:
                print(f"[CPU] Preprocessing error: {e}")
                time.sleep(0.2)

    def preprocess_gpu(self):
        while self.thread_running.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5)
                print("Frame received for preprocessing [GPU].")

                preprocessed_frame = preprocess_frame(frame)
                print("Preprocessing Complete [GPU]")

                if preprocessed_frame.size > 0:
                    self.preprocessed_queue.put(preprocessed_frame, timeout=0.5)
                else:
                    print("Preprocessed frame invalid [GPU].")

            except Exception as e:
                print(f"[GPU] Preprocessing error: {e}")
                time.sleep(0.2)

    def stop(self):
        if self.use_gpu:
            self.thread_running.clear()
            if self.thread_worker and self.thread_worker.is_alive():
                self.thread_worker.join(timeout=2)
                if self.thread_worker.is_alive():
                    self.thread_worker.terminate()
        else:
            self.process_running.clear()
            for p in self.processes:
                if p and p.is_alive():
                    p.join(timeout=2)
                    if p.is_alive():
                        p.terminate()
            

