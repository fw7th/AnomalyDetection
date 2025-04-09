from src.prediction.frame_read import Frames
from src.prediction.process_frame import FrameProcessor_
from src.prediction.detection import ObjectDetector
from src.prediction.tracking import ObjectTracker
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
        self.event1 = Event()

        if self.use_gpu:
            self.event2 = Event()
            self.event3 = Event()
            self.event4 = Event()
        else:
            self.manager = Manager()
            self.event2 = self.manager.Event()
            self.event3 = self.manager.Event()
            self.event4 = self.manager.Event()

        self.event5 = Event()


        # Core modules
        self.framing = Frames(self.source, self.event1)
        self.preprocessing = FrameProcessor_(self.framing.frame_queue)
        self.detect = ObjectDetector(self.preprocessing.preprocessed_queue)
        self.track = ObjectTracker(self.detect.detection_queue)
        self.display = VideoDisplay(
            self.track.tracker_queue,
            enable_saving=self.enable_saving,
            save_dir=self.save_dir
        )

    def frame_create(self):
        self.framing.read_frames()
        self.event1.set()

    def preprocess_frame(self, event1, event2):
        print("[EVENT1] Waiting for event1...")
        event1.wait()
        print("[EVENT1] Event1 is not set!.")

        if self.use_gpu:
            self.preprocessing.preprocess_gpu()
        else:
            self.preprocessing.preprocess_cpu()

        event2.set()

    def detect_on_frame(self, event2, event3):
        event2.wait()
        if self.use_gpu:
            self.detect.gpu_track()
        else:
            self.detect.cpu_track()
        event3.set()

    def track_detections(self, event3, event4):
        event3.wait()
        self.track._detect_humans()
        event4.set()

    def display_frames(self, event4, event5):
        event4.wait()
        self.display.display_video()
        event5.set()

    def setup_and_start(self):
        self.thread_processes = []
        self.detect_processes = []
        self.track_processes = []

        self.frame_thread = Thread(target=self.frame_create, daemon=True)

        if self.use_gpu:
            self.prep_thread = Thread(
                target=self.preprocess_frame,
                args=(self.event1, self.event2),
                daemon=True
            )
            self.detect_thread = Thread(
                target=self.detect_on_frame,
                args=(self.event2, self.event3),
                daemon=True
            )
            self.track_thread = Thread(target=self.track_detections,
                args=(self.event3, self.event4),
                daemon=True
            )

        else:
            for _ in range(2):
                pre = Process(
                    target=self.preprocess_frame,
                    args=(self.event1, self.event2),
                    daemon=True
                )
                self.thread_processes.append(pre)

                det = Process(
                    target=self.detect_on_frame,
                    args=(self.event2, self.event3),
                    daemon=True
                )
                self.detect_processes.append(det)
                
                tra = Process(
                    target=self.track_detections,
                    args=(self.event3, self.event4),
                    daemon=True
                )
                self.track_processes.append(tra)
        
        self.display_thread = Thread(
            target=self.display_frames,
            args=(self.event4, self.event5),
            daemon=True
        )

    def start(self):
        self.frame_thread.start()
        if self.use_gpu:
            self.prep_thread.start()
            self.detect_thread.start()
            self.track_thread.start()

        else:
            for p in self.thread_processes:
                p.start() 
            for k in self.detect_processes:
                k.start()
            for z in self.track_processes:
                z.start()

        self.display_thread.start()

    def stop(self):
        self.frame_thread.join()
        if self.use_gpu:
            self.prep_thread.join()
            self.detect_thread.join()
            self.track_thread.join()

        else:
            for p in self.thread_processes:
                if p and p.is_alive():
                    p.join() 
            for k in self.detect_processes:
                if k and k.is_alive():
                    k.join() 
            for z in self.track_processes:
                if z and z.is_alive():
                    z.join()

        self.display_thread.join()
        
        # Clear events
        self.event1.clear()
        self.event2.clear()
        self.event3.clear()
        self.event4.clear()
        self.event5.clear()

    def run(self):
        self.setup_and_start()
        self.start()
        self.stop()
