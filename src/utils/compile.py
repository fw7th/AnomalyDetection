from src.prediction.process_frame import FrameProcessor_
from src.utils.display import VideoDisplay
from threading import Thread, Event
from multiprocessing import Process, Manager
import torch.cuda

class Compile:
    def __init__(self, source, enable_saving=False, save_dir=None):
        # Essentials
        self.use_gpu = torch.cuda.is_available()
        self.source = source
        self.save_dir = save_dir
        self.enable_saving = enable_saving

        # Events
        if self.use_gpu:
            self.event2 = Event()
            self.event3 = Event()
            self.event4 = Event()
            self.event5 = Event()
        else:
            self.manager = Manager()
            self.event2 = self.manager.Event()
            self.event3 = self.manager.Event()
            self.event4 = self.manager.Event()
            self.event5 = self.manager.Event()

        # Core modules
        self.preprocessing = FrameProcessor_(self.source)
        self.detect = self.preprocessing.detector
        self.track = self.detect.tracker
        self.display = VideoDisplay(
            enable_saving=self.enable_saving,
            save_dir=self.save_dir
        )

    def preprocess_frame(self):
        if self.use_gpu:
            self.preprocessing.thread_running.set()
            self.preprocessing.preprocess_gpu()
        else:
            self.preprocessing.process_running.set()
            self.preprocessing.preprocess_cpu()

        self.event2.set()

    def detect_on_frame(self):
        self.event2.wait(timeout=0.1)
        self.detect.thread_running.set() if self.use_gpu else self.detect.process_running.set()
        self.detect._detect_humans()
        self.event3.set()

    def track_detections(self):
        self.event3.wait(timeout=0.1)
        self.track.gpu_track() if self.use_gpu else self.track.cpu_track()
        self.track.thread_running.set() if self.use_gpu else self.track.process_running.set()
        self.event4.set()

    def display_frames(self):
        self.event4.wait(timeout=0.2)
        self.display.running.set()
        self.display.display_video()
        self.event5.set()

    def setup_and_start(self):
        self.thread_processes = []
        self.detect_processes = []
        self.track_processes = []


        if self.use_gpu:
            self.prep_thread = Thread(target=self.preprocess_frame,)
            self.detect_thread = Thread(target=self.detect_on_frame,)
            self.track_thread = Thread(target=self.track_detections,)

        else:
            self.pre = Process(target=self.preprocess_frame,)
            self.det = Process(target=self.detect_on_frame,)
            self.tra = Process(target=self.track_detections,)

        self.display_thread = Thread(target=self.display_frames,)

    def start(self):
        if self.use_gpu:
            self.prep_thread.start()
            self.detect_thread.start()
            self.track_thread.start()

        else:
            self.pre.start() 
            self.det.start()
            self.tra.start()

        self.display_thread.start()

    def stop(self):
        if self.use_gpu:
            self.preprocessing.thread_running.clear()
            self.detect.thread_running.clear()
            self.prep_thread.join()
            self.detect_thread.join()
            self.track_thread.join()

        else:
            self.preprocessing.process_running.clear()
            self.detect.process_running.clear()
            if self.pre and self.pre.is_alive():
                self.pre.join() 
            if self.det and self.det.is_alive():
                self.det.join() 
            if self.tra and self.tra.is_alive():
                self.tra.join()

        self.display_thread.join()
        
        # Clear events
        self.event2.clear()
        self.event3.clear()
        self.event4.clear()
        self.event5.clear()

    def run(self):
        self.setup_and_start()
        self.start()
        self.stop()
