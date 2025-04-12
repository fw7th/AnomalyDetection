import cv2 as cv
import threading
from multiprocessing import Queue

class Frames:
    def __init__(self, source):
        self.source = source
        self.lock = threading.Lock()
        self.frame_queue = Queue(maxsize=20)
        self.running = threading.Event()
        self.thread = None
        self.cap = None

    def read_frames(self): 
        if self.source is None:
            print("Error: No video source provided.")
            return 
       
        self.cap = cv.VideoCapture(self.source)
        if not self.cap.isOpened():
            print("Error: Failed to open video source.")
            return
        
        print("Frames being pulled from source")

        try:
            while self.running.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Source isn't True")
                    break
                
                self.frame_queue.put(frame)

        finally: 
            self.cap.release()

    def stop_reading(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
