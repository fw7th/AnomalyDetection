"""
detection.py


Provides functionality for multithreaded detection of humans.
Includes classes and functions for handling detection, annotation, multithreading,
and seperate return class funtions.


Author: fw7th
Date: 2025-03-20
"""

from ultralytics import YOLO
import threading
import supervision as sv
from src import config
import cv2 as cv

CLASS = 0  # Class 0 is the human class in YOLO.
VID_STRIDE = 1  # Skip every one frame for more efficient processing.

class ObjectDetector:
    """
    Multithreaded implementation to detect humans only, ensuring thread safety

    ...

    Attributes
    ----------
    None


    Methods
    -------
    threaded_capture(source=None)
        Takes a source and starts a thread for image processing.

    _object_generator()
        Detects humans and annotates frames. 
        
    get_latest_results()
        Continuously returns detections and frames in real time through multithreading.

    stop()
        Stops the threading and ends the detection algorithm.
    """


    def __init__(self):
        """
        Initializes ObjectDetector instance.


        Parameters
        ----------
        model : -- 
            gotten from the config module, detector model used is yolo11n.
        frame : numpy.ndarray
            stores every frame gotten from the model in real time.
        results : numpy.ndarray
            stores the detections and their positions from the YOLO model.
        detections : numpy.ndarray             
            used to process results using supervison.
        annotated : numpy.ndarray
            stores annotated frames from the predictions.
        running : bool
            processing state of multithreading operation.
        lock : 
            Used to ensure thread safety and avoid race conditions.
        """

        self.model = YOLO(config.MODEL_PATH)
        self.frame = None
        self.results = None
        self.detections = None
        self.annotated = None
        self.running = False
        self.lock = threading.Lock() # Thread safety


    def threaded_capture(self, source=None):
        """ Method detects object efficiently through multithreading.


        Parameters
        ----------
        source : str, int
            Link/source of the video stream or static video file.


        Returns
        -------
        self.cap : numpy.ndarray
            Shows if the source was opened or not, returns frames of the file if True.
        self : ----
            Returns ObjectDetector class objects.
        """

        if source is None:
            print("Error: No video source provided.")
            return None, None

        self.cap = cv.VideoCapture(source)
        if not self.cap.isOpened():
            print("Error: Failed to open video source.")

        
        self.running = True
        self.thread = threading.Thread(target=self._object_generator, daemon=True)
        self.thread.start()

        return self.cap, self  # Return cap & 


    def _object_generator(self):
        """ Runs detection in a separate thread while updating detections.

        Uses multithreading to detect, process and annotate humans in a frame.
        """
        
        from src.utils.general import BOX_ANNOTATOR
        while self.running:
            success, frame = self.cap.read()
            if not success:
                break

            results = self.model(
                frame,
                classes=CLASS,
                vid_stride=VID_STRIDE
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            annotated = BOX_ANNOTATOR.annotate(frame.copy(), detections=detections)

            # Store the latest results with thread safety
            with self.lock:
                self.frame = frame
                self.detections = detections
                self.annotated = annotated
                self.results = results

        self.cap.release()
        cv.destroyAllWindows()


    def get_latest_results(self):
        """
        Safely get the latest detection results.
        
        Returns
        -------
        self.results : numpy.ndarray
            latest detections and positions from YOLO model.
        self.annotated : numpy.ndarray
            latest annotated frame showing detected humans.
        self.detections : numpy.ndarray
            preprocsssing results from supervision.
        """

        with self.lock:
            return self.detections, self.annotated, self.results


    def stop(self):
        """Stops the detection loop and safely ends thread."""
        self.running = False
        self.thread.join()
