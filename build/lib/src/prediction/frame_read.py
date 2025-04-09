import cv2 as cv
import threading
import queue
import time

class Frames:
    def __init__(self, source):
        self.source = source
        self.frame_queue = queue.Queue(maxsize=15)
        self.lock = threading.Lock()
        self.running = threading.Event()
        self.thread = None
        self.cap = None

    def read_frames(self): 
        if self.source is None:
            print("Error: No video source provided.")
            self.running.clear()
            return 
       
        self.cap = cv.VideoCapture(self.source)
        if not self.cap.isOpened():
            print("Error: Failed to open video source.")
            self.running.clear()
            return

        try:
            while not self.running.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Source isn't True")
                    time.sleep(0.1)
                    break
                
                print("Frames being pulled")

                try:
                    self.frame_queue.put(frame.copy(), timeout=0.1)
                    print("Frames added to frame queue")

                except queue.Full:
                    print("Frame preprocessing is slow! 1")
                    time.sleep(0.1)

                time.sleep(0.01)
        finally: 
            self.cap.release()

    def start_reading(self):
        self.running.set()
        self.thread = threading.Thread(target=self.read_frames, daemon=True)
        self.thread.start()

    def stop_reading(self):
        self.running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.cap:
            self.cap.release()
