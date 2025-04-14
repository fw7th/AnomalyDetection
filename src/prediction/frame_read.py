import cv2 as cv
import threading

class Frames:
    def __init__(self, frame_queue, source):
        self.source = source
        self.lock = threading.Lock()
        self.frame_queue = frame_queue
        self.running = threading.Event()
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

        while self.running.is_set():
            try:
                print("Trying to pull frames from video")
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Source isn't True")
                    break
                
                with self.lock:
                    self.frame_queue.put(frame)

            except Exception as e:
                print(f"Error with frame reading: {e}")

            finally: 
                self.cap.release()
