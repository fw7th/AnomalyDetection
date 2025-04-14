import multiprocessing as mp
from ultralytics import YOLO
from src.config.config_env import V8_PATH
import numpy as np
import cv2

def run_yolo(frame, result_queue):
    print("Loading model in process...")
    model = YOLO(V8_PATH)
    print("Model loaded, running inference...")
    results = model(frame, classes=[0], conf=0.25, stream=True)
    print("Inference complete!")
    result_queue.put(results)

if __name__ == "__main__":
    # Create a dummy frame
    frame = "/home/fw7th/Videos/1.mp4"
    # Or load an actual image
    # frame = cv2.imread("test_image.jpg")
    
    result_queue = mp.Queue()
    p = mp.Process(target=run_yolo, args=(frame, result_queue))
    print("Starting process...")
    p.start()
    p.join(timeout=60)  # Wait up to 60 seconds
    
    if p.is_alive():
        print("Process still running after timeout - possible hang!")
        p.terminate()
    else:
        print("Process completed normally")
        if not result_queue.empty():
            results = result_queue.get()
            print(f"Got results: {type(results)}")
