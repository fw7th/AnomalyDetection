from src.utils.preprocessing import preprocess_frame
from src.prediction.frame_read import Frames
from src.prediction.detection import ObjectDetector
import queue as std_queue
import multiprocessing as mp
import threading
import torch
import time

class FrameProcessor_:
    def __init__(self, source, maxsize=25):
        self.source = source
        self.use_gpu = torch.cuda.is_available()
        self.frame_reader = Frames(self.source)
        self.detector = ObjectDetector()

        if self.use_gpu:
            self.preprocessed_queue = std_queue.Queue(maxsize)
            self.thread_running = threading.Event()
        else:
            self.detector.preprocess_recv, self.preprocess_send  = mp.Pipe(duplex=False)
            self.process_running = mp.Event()

    def start_frame_reader(self):
        self.frame_reader.running.set()
        self.frame_thread = threading.Thread(target=self.frame_reader.read_frames, daemon=True)
        self.frame_thread.start()

    def stop_frame_reader(self):
        self.frame_reader.running.clear()
        if self.frame_thread and self.frame_thread.is_alive():
            self.frame_thread.join()

    def preprocess_cpu(self):
        self.start_frame_reader()
        try:
            while self.process_running.is_set():
                frame = self.frame_reader.frame_queue.get(timeout=0.1)
                print("Frame received for preprocessing [CPU].")

                preprocessed_frame = preprocess_frame(frame)
                print("Preprocessing Complete [CPU]")

                if preprocessed_frame.size > 0:
                    self.preprocess_send.send(preprocessed_frame)
                    print("Frame sent to detect_pipe") 
                else:
                    print("Preprocessed frame invalid [CPU].")

        except Exception as e:
            print(f"[CPU] Preprocessing error: {e}")
            time.sleep(0.2)
        
        finally:
            self.stop_frame_reader()

    def preprocess_gpu(self):
        self.start_frame_reader()
        while self.thread_running.is_set():
            try:
                frame = self.frame_reader.frame_queue.get(timeout=0.5)
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

        self.stop_frame_reader()

    def stop(self):
        if self.use_gpu:
            self.thread_running.clear()
            if self.thread_worker and self.thread_worker.is_alive():
                self.thread_worker.join(timeout=2)
        else:
            self.process_running.clear()
            for p in self.processes:
                if p and p.is_alive():
                    p.join(timeout=2)
                    if p.is_alive():
                        p.terminate()
            

