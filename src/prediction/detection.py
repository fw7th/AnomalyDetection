"""
detection.py

Provides functionality for multithreaded human detection.
This module includes classes and functions for handling detection, annotation, 
and multithreading while ensuring thread safety.

Author: fw7th
Date: 2025-03-20
"""

from ultralytics import YOLO
import threading
import supervision as sv
from src.config import config_env
import cv2 as cv
import time

CLASS = 0  # Class 0 corresponds to humans in YOLO.

class ObjectDetector:
    """
    A multithreaded YOLO-based human detection system.

    This class provides a real-time, thread-safe approach to detecting humans 
    in a video stream using YOLO and Supervision.

    Attributes
    ----------
    (Defined in the constructor docstring)
       
    Methods
    -------
    threaded_capture(self.source=None)
        Starts a video stream and spawns a thread for human detection.

    _object_generator()
        Continuously detects and annotates frames in a separate thread.

    get_latest_results()
        Retrieves the latest detection results in a thread-safe manner.

    stop()
        Stops the detection loop and safely terminates the thread.
    """

    def __init__(self, source):
        """
        Initializes an ObjectDetector instance.

        Attributes
        ----------
        model : YOLO
            YOLO detection model loaded from the configuration file.
        frame : numpy.ndarray or None
            Stores the most recent video frame.
        results : numpy.ndarray or None
            Stores the detections and their positions from YOLO.
        detections : sv.Detections or None
            Stores processed detections.
        annotated : numpy.ndarray or None
            Stores the annotated frame with detected objects.
        running : bool
            Indicates whether the detection thread is active.
        lock : threading.Lock
            Ensures thread safety when accessing shared attributes.
        source : str or int
            source of video stream.
        """
        self.model = YOLO(config_env.MODEL_PATH, task='detect', verbose=False)
        self.frame = None
        self.results = None
        self.detections = None
        self.annotated = None
        self.running = False
        self.lock = threading.Lock()  # Ensures thread safety
        self.source = source 

    def threaded_capture(self):
        """
        Starts the object detection process in a separate thread.

        Returns
        -------
        cap : cv.VideoCapture
            OpenCV video capture object.
        self : ObjectDetector
            The current instance of the detector.
        """
        if self.source is None:
            print("Error: No video source provided.")
            return None, None

        self.cap = cv.VideoCapture(self.source)
        if not self.cap.isOpened():
            print("Error: Failed to open video source.")
            return None, None

        self.running = True
        self.thread = threading.Thread(target=self._object_generator, daemon=True)
        self.thread.start()

        return self.cap, self

    def _object_generator(self):
        """
        Runs detection in a separate thread while updating results.

        This method:
        - Captures frames from the video source.
        - Applies YOLO detection to identify humans.
        - Uses Supervision for preprocessing and annotation.
        - Updates the latest processed results in a thread-safe manner.

        Updates
        -------
        self.frame : numpy.ndarray
            The latest video frame.
        self.results : numpy.ndarray
            YOLO detection results.
        self.detections : sv.Detections
            Processed detection outputs.
        self.annotated : numpy.ndarray
            The latest annotated frame with detections.
        """
        # Import the annotator of our choice.
        from src.utils.general import CORNER_ANNOTATOR
        
        # Import the zones from the JSON file. 
        from src.utils.general import load_zones_config

        # Load polygons from our source video.
        polygons = load_zones_config(config_env.POLYGONS)
        zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,)
            )
            for polygon in polygons
        ]

        while self.running:
            success, frame = self.cap.read()
            if not success:
                # For streams, there might be connection loss, let's add a small timeout.
                time.sleep(0.1)
                continue
            
            # Preprocess and detect
            results = self.model(
                frame,
                classes=CLASS,
            )[0]
            detections = sv.Detections.from_ultralytics(results)

            annotated = frame.copy()
            # Loop through the polygon and draw it on each frame
            for idx, zone in enumerate(zones):
                annotated = sv.draw_polygon(
                    scene=annotated,
                    polygon=zone.polygon,
                    thickness=2
                )
                detections = detections[zone.trigger(detections)]
                annotated = CORNER_ANNOTATOR.annotate(annotated, detections=detections)
            # Store the latest results safely
            with self.lock:
                self.frame = frame
                self.detections = detections
                self.annotated = annotated
                self.results = results

        self.cap.release()
        cv.destroyAllWindows()

    def get_latest_results(self):
        """
        Retrieves the latest detection results in a thread-safe manner.

        Returns
        -------
        detections : sv.Detections or None
            Processed detection outputs.
        annotated : numpy.ndarray or None
            The latest annotated frame with detections.
        results : numpy.ndarray or None
            Raw YOLO detection results.
        """
        with self.lock:
            return self.detections, self.annotated, self.results

    def stop(self):
        """
        Stops the detection process and safely terminates the thread.

        Ensures that the detection loop exits cleanly, releasing resources.
        """
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()
