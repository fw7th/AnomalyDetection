import cv2 as cv
import threading
from multiprocessing import Queue
import queue
import time

class Frames:
    def __init__(self, source, first_frame_event):
        self.source = source
        self.frame_queue = Queue(maxsize=15)
        self.lock = threading.Lock()
        self.running = True  # Just a simple bool now
        self.first_frame_event = first_frame_event
        self.first_frame_pushed = False
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

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Source isn't True")
                    time.sleep(0.1)
                    break
                
                print("Frames being pulled")

                try:
                    self.frame_queue.put(frame.copy(), timeout=0.1)
                    print("Frames added to frame queue")

                    if self.first_frame_event and not self.first_frame_pushed:
                        print("Setting event for the first frame")
                        self.first_frame_event.set()  # Signal that frame queue is populated
                        self.first_frame_pushed = True

                except queue.Full:
                    print("Frame preprocessing is slow! 1")
                    time.sleep(0.1)

                time.sleep(0.01)
        finally: 
            self.cap.release()

    def stop_reading(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.cap:
            self.cap.release()
