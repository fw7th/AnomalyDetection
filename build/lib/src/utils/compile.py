"""
compile.py

A comprehensive pipeline for human detection and tracking with intelligent
reself.source management, proper initialization/shutdown, and better error handling.

Author: fw7th
Date: 2025-04-04
"""

from src.prediction.frame_read import Frames
from src.prediction.process_frame import FrameProcessor_
from src.prediction.detection import ObjectDetector
from src.prediction.tracking import ObjectTracker
from src.utils.display import VideoDisplay
from threading import Thread, Event
import multiprocessing as mp
import torch.cuda

class Compile:
    def __init__(self, source, enable_saving=False, save_dir=None):
        self.use_gpu = torch.cuda.is_available()
        self.source = source
        self.save_dir = save_dir
        self.enable_saving = enable_saving
        self.framing = Frames(self.source)
        self.preprocessing = FrameProcessor_(self.framing.frame_queue)
        self.detect = ObjectDetector(self.preprocessing.preprocessed_queue)
        self.track = ObjectTracker(self.detect.detection_queue)
        self.display = VideoDisplay(
            self.track.tracker_queue,
            enable_saving=self.enable_saving,
            save_dir=self.save_dir
        )
        self.event1_ready = Event()

        if self.use_gpu:
            self.event2_ready = Event()
            self.event3_ready = Event()
            self.event4_ready = Event()
        else:
            self.event2_ready = mp.Event()
            self.event3_ready = mp.Event()
            self.event4_ready = mp.Event()
        
        self.event5_ready = Event()

    def frame_create(self):
        self.framing.read_frames()
        self.event1_ready.set()

    def preprocess_frame(self):
        self.event1_ready.wait()
        if self.use_gpu:
            self.preprocessing.preprocess_gpu()
            self.event2_ready.set()
        
        else:
            self.preprocessing.preprocess_cpu()
            self.event2_ready.set()

    def detect_on_frame(self):
        self.event2_ready.wait()
        if self.use_gpu:
            self.detect.gpu_track()
            self.event3_ready.set()

        else:
            self.detect.cpu_track()
            self.event3_ready.set()

    def track_detections(self):
        self.event3_ready.wait()
        if self.use_gpu:
            self.track._detect_humans()
            self.event4_ready.set()

        else:
            self.track._detect_humans()
            self.event4_ready.set()
    
    def display_frames(self):
        self.event4_ready.wait()
        self.display.display_video()
        self.event5_ready.set()
  
    def TandP(self):
        self.thread_processes = []
        self.detect_processes = []
        self.track_processes = []

        self.frame_thread = Thread(target=self.frame_create, daemon=True)

        if self.use_gpu:
            self.preP_thread = Thread(target=self.preprocess_frame, daemon=True)
            self.detect_thread = Thread(target=self.detect_on_frame, daemon=True)
            self.track_thread = Thread(target=self.track_detections, daemon=True)

        else:
            for _ in range(2):
                pre = mp.Process(target=self.preprocess_frame,)
                self.thread_processes.append(pre)

                det = mp.Process(target=self.detect_on_frame, args=(_,))
                self.detect_processes.append(det)
                
                tra = mp.Process(target=self.track_detections,)
                self.track_processes.append(tra)
        
        self.display_thread = Thread(target=self.display_frames, daemon=True)

    def start(self):
        self.frame_thread.start()

        if self.use_gpu: 
            self.preP_thread.start() 
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


    def stop_all(self):
        self.frame_thread.join()
        if self.use_gpu:
            self.preP_thread.join()
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


    def run(self):
        self.start()
        self.stop_all()
