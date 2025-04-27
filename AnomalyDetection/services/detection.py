"""
detection.py

Provides optimized functionality for multithreaded human detection.
This module includes classes and functions for handling detection, annotation, 
and multithreading while ensuring thread safety.

Author: fw7th
Date: 2025-04-02
"""

from utils import messaging_system, visual_alerts, sound_alerts
from config import settings 
from core import LOG, load_zones_config
from ultralytics.utils.ops import LOGGER
from ultralytics import YOLO
import multiprocessing as mp
import supervision as sv
import numpy as np
import time, os, queue, torch, logging
import threading

# Set PyTorch thread count to optimize for 2 cores
torch.set_num_threads(2)

class ObjectDetector:
    """
    Multithreaded YOLO-based human detection system.

    This class provides a real-time, thread-safe approach to detecting humans 
    in a video stream using YOLO and Supervision libraries.

    Attributes:
        use_gpu (bool): Indicates if CUDA is available for GPU acceleration.
        last_log_time (float): Last time FPS was logged.
        frame_count (int): Number of processed frames since last log.
        alert_active (bool): Whether an alert is currently active.
        cooldown (bool): Whether the alert system is in cooldown.
        alert_start_time (float): Timestamp when the current alert started.
        cooldown_start_time (float): Timestamp when cooldown started.
        last_beep_time (float): Last time the sound alert was triggered.
        beep_interval (int): Minimum interval between sound alerts (in seconds).
        last_message_time (float): Last time a message was sent.
        message_interval (int): Minimum interval between alert messages (in seconds).
        message_active (bool): Whether a message alert is active.
        message_cooldown (bool): Whether message sending is cooling down.
        message_cooldown_start (float): Timestamp when message cooldown started.
        detections (sv.Detections): Latest detection results.
        results (ultralytics.engine.results.Results): Latest YOLO inference results.
        frame (np.ndarray): Current frame being processed.
        buffer (list): Frame buffer for temporary storage.
        box_annotator (sv.RoundBoxAnnotator): Annotator for drawing detection boxes.
        model (YOLO): YOLO object detection model.
        zones (list): List of polygon zones used for restricted area detection.
        preprocessed_queue (multiprocessing.Queue): Input queue for preprocessed frames.
        detection_queue (multiprocessing.Queue): Output queue for processed frames.
        your_num (str): Phone number for sending alert messages (optional).
        your_mail (str): Email address for sending alert messages (optional).
        messenger (messaging_system): Messaging system instance for alerts.
        _running (multiprocessing.Event): Flag to control running state.
    """

    def __init__(self, preprocessed_queue, detection_queue, your_num=None, your_mail=None, accuracy=None):
        """
        Initializes the ObjectDetector with required queues and optional alert settings.

        Args:
            preprocessed_queue (multiprocessing.Queue): Queue containing preprocessed frames.
            detection_queue (multiprocessing.Queue): Queue to send processed frames.
            your_num (str, optional): Phone number to send SMS alerts.
            your_mail (str, optional): Email address to send email alerts.
            accuracy(str, int, optional): Used to select the accuracy of inference.
        """

        self.use_gpu = torch.cuda.is_available()
        self.accuracy = accuracy
        self.last_log_time = time.time()
        self.frame_count = 0
        self.alert_active = None
        self.cooldown = None
        self.alert_start_time = 0
        self.cooldown_start_time = 0
        self.last_beep_time = 0
        self.beep_interval = 1

        self.last_message_time = 0
        self.message_interval = 240
        self.message_active = None
        self.message_cooldown = None
        self.message_cooldown_start = 0
        self.detections = None
        self.results = None
        self.frame = None
        self.buffer = []

        # Initialize annotator as instance variable
        self.box_annotator = sv.RoundBoxAnnotator(thickness=1)

        self.model = None
        self.zones = self._initialize_zones()

        self.preprocessed_queue = preprocessed_queue
        self.detection_queue = detection_queue
        
        self.your_num = your_num
        self.your_mail = your_mail
        self.messenger = messaging_system(self.your_num, self.your_mail)
        """
        if self.use_gpu:
            self._running = threading.Event()
        """
        self._running = mp.Event()

    def _initialize_model(self):
        """
        Initializes the YOLO model for detection with minimal console output.

        Returns:
            YOLO: Initialized YOLO model instance or None if loading fails.
        """
        # Silence YOLO's logger before loading the model
        LOGGER.setLevel(logging.ERROR)  # Only show errors, not info messages

        # Also silence the underlying libraries
        os.environ['PYTHONIOENCODING'] = 'utf-8'  # Ensures proper encoding for suppressed output
        os.environ['ULTRALYTICS_SILENT'] = 'True'  # Specifically for ultralytics
        
        try:
            # Load model with best device selection
            if self.accuracy == 3 or self.accuracy == "high":
                model = YOLO(settings.V8_PATH)
                LOG.info("Utilizing max accuracy")

            elif self.accuracy == 1 or self.accuracy == "low":
                model = YOLO(settings.VINO_PATH)
                LOG.info("Utilizing low accuracy")

            elif self.accuracy == 2 or self.accuracy == "mid" or self.accuracy is None:
                model = YOLO(settings.ONNX_PATH)
                LOG.info("Utilizing mid accuracy")

            return model
        
        except ValueError as e:
            LOG.error(f"Invalid input: {e}. Please enter a valid option or leave as None.")
        except Exception as e:
            LOG.critical(f"Error initializing model: {e}")
            return None

    def _initialize_zones(self):
        """
        Initializes detection zones from the configured polygons.

        Returns:
            list: List of Supervision PolygonZone instances.
        """
        try:
            polygons = load_zones_config(settings.POLYGONS)
            return [
                sv.PolygonZone(
                    polygon=polygon,
                    triggering_anchors=(sv.Position.CENTER,)
                )
                for polygon in polygons
            ]
        except Exception as e:
            LOG.error(f"Error initializing zones: {e}")
            return []

    def detect_single_frame(self):
        """
        Processes a single frame for human detection.

        - Retrieves a frame from the preprocessed queue.
        - Runs YOLO detection.
        - Filters detections within specified zones.
        - Annotates the frame with bounding boxes.
        - Sends the processed frame to the detection queue.
        - Triggers alert and messaging systems based on detections.
        """
        try:
            # Get frame from queue with timeout
            self.frame = self.preprocessed_queue.get(timeout=0.001)
        
        except queue.Empty:
            time.sleep(0.01)
            LOG.warning("Waiting for preprocessed frames... bottleneck!")
            return

        except Exception as e:
            LOG.warning(f"Error in detection frame extraction from preprocessing: {e}")
            time.sleep(0.01)
            return
        
        # Run detection on this frame
        try:
            # First check if frame is valid
            if self.frame is None or not isinstance(self.frame, np.ndarray) or self.frame.size == 0:
                LOG.error("ERROR: Invalid frame for detection")
                return
                
            # Check model initialization
            if self.model is None:
                LOG.warning("Model is not initialized")
                self.model = self._initialize_model()
                if self.model is None:
                    LOG.critical("Model could not be initialized after multiple attempts, program ending")
                    return

            try:
                self.results = self.model(
                    self.frame,
                    classes=[0],  # Person class
                    conf=0.35,    # Higher confidence threshold for fewer false positives
                    iou=0.45,     # Lower IOU threshold for better performance
                    device="cuda" if self.use_gpu else "cpu",
                    augment=False,
                    agnostic_nms=True,
                    task="detect",
                    verbose=False
                )[0]

            except Exception as e:
                import traceback
                LOG.critcal(f"""
                ===== YOLO DETECTION ERROR =====
                Error type: {type(e).__name__}
                Error message: {str(e)}
                ------- Stack trace: -------
                {traceback.format_exc()}
                ===============================
                """)
                return 
            
        except Exception as e:
            import traceback
            LOG.error(f"Outer detection loop error: {e}")
            LOG.error(traceback.format_exc())
            time.sleep(0.01)
            return

        try:
            # Convert YOLO results to Supervision detections
            self.detections = sv.Detections.from_ultralytics(self.results)
        except Exception as e:
            LOG.critical("ERROR in Supervision conversion!")
            import traceback
            LOG.error(f"Error creating detections: {e}")
            LOG.error(traceback.format_exc())
            return
    
        # Process valid frame with detections
        if self.detections is not None:
            processed_frame = self.frame.copy()
            
            # Process zones if they exist
            if self.zones:
                # Draw all zones
                for idx, zone in enumerate(self.zones):
                    processed_frame = sv.draw_polygon(
                        scene=processed_frame,
                        polygon=zone.polygon,
                        thickness=2
                    )
                
                # Create a mask for combined zone filtering
                combined_mask = np.zeros(len(self.detections), dtype=bool)
                if len(self.detections) > 0 and self.detections:
                    for zone in self.zones:
                        # Use OR to combine results from all zones
                        zone_mask = zone.trigger(self.detections)
                        combined_mask = np.logical_or(combined_mask, zone_mask)
                    
                    # Apply the combined filter
                    if np.any(combined_mask):
                        self.detections = self.detections[combined_mask]
                    else:
                        # Keep original detections if no zones triggered
                        pass
            
                    # Add detections visualization if any detections remain
                    processed_frame = self.box_annotator.annotate(
                        scene=processed_frame, 
                        detections=self.detections
                    )
            
            # Log FPS periodically using the class frame counter
            self.frame_count += 1
            if time.time() - self.last_log_time >= 5.0:  # Log every 4 seconds
                fps = self.frame_count / (time.time() - self.last_log_time)
                LOG.info(f"Inference at {fps:.2f} FPS")
                self.frame_count = 0
                self.last_log_time = time.time()

            if self.detections:
                self.message_active = True
                self.message_system()
                self.alert_system(processed_frame)

            else:
                self.message_active = False

            try:
                # Send the processed frame to the output queue
                self.detection_queue.put(processed_frame, timeout=0.001)
                
                # Force garbage collection periodically
                if self.frame_count % 20 == 0:  # Less frequent GC
                    import gc 
                    gc.collect()
                    if self.use_gpu:
                        torch.cuda.empty_cache()
                    LOG.info(f"Memory cleaned after {self.frame_count} frames")

            except queue.Full:
                time.sleep(0.001)

            except Exception as e:
                LOG.critical(f"Error: Failed to send processed frame to queue: {e}, exiting program")
                return

    def alert_system(self, processed_frame):
        """
        Handles visual and sound alerts based on active detections.

        Args:
            processed_frame (np.ndarray): Frame to apply visual alerts on.
        """
        if not self.alert_active:
            self.alert_active = True
            self.alert_start_time = time.time()

        if self.alert_active:
            if time.time() - self.alert_start_time < 7:
                processed_frame = visual_alerts(processed_frame)

                # Only play beep every 1.5 secs
                if time.time() - self.last_beep_time >= self.beep_interval:
                    self.last_beep_time = time.time()
                    sound_alerts()

            elif not self.cooldown:
                self.cooldown = True
                self.cooldown_start_time = time.time()

        if self.cooldown:
            if time.time() - self.cooldown_start_time >= 3:
                self.alert_active = False
                self.cooldown = False

    def message_system(self):
        """
        Handles sending alert messages through configured communication channels (SMS, email).
        """
        if self.message_active:
            current_time = time.time()

            if not self.message_cooldown:
                if self.your_num and not self.your_mail:
                    threading.Thread(target=self.messenger.send_twilio_message, daemon=True).start()
                    LOG.info(f"Alert message sent to {self.your_num}.")

                elif self.your_mail and not self.your_num:
                    threading.Thread(target=self.messenger.send_email, daemon=True).start()
                    LOG.info(f"Alert message sent to {self.your_mail}.")

                elif not self.your_mail and not self.your_num:
                    LOG.info("No mail or number to send messages to")

                else:
                    threading.Thread(target=self.messenger.send_to_both, daemon=True).start()
                    LOG.info(f"Alert message sent to {self.your_num} and {self.your_mail}.")

                self.message_cooldown = True
                self.message_cooldown_start = current_time
            
            # Check if cooldown period is over
            elif current_time - self.message_cooldown_start > self.message_interval:
                self.message_cooldown = False  # Allow next message
