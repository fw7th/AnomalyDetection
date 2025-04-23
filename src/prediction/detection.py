
"""
detection.py

Provides optimized functionality for multithreaded human detection.
This module includes classes and functions for handling detection, annotation, 
and multithreading while ensuring thread safety.

Author: fw7th
Date: 2025-04-02
"""

from src.utils.general import load_zones_config  # Import the zones from the JSON file
from src.utils.alert_system import visual_alerts, sound_alerts
from src.utils.messaging import send_twilio_message
from src.config import config_env
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
    A multithreaded YOLO-based human detection system.

    This class provides a real-time, thread-safe approach to detecting humans 
    in a video stream using YOLO and Supervision.
    """

    def __init__(self, preprocessed_queue, detection_queue):
        """
        Initializes an ObjectDetector instance with improved performance.
        """

        self.use_gpu = torch.cuda.is_available()
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
        
        """
        if self.use_gpu:
            self._running = threading.Event()
        """
        self._running = mp.Event()

    def _initialize_model(self):
        # Silence YOLO's logger before loading the model
        LOGGER.setLevel(logging.ERROR)  # Only show errors, not info messages

        # Also silence the underlying libraries
        os.environ['PYTHONIOENCODING'] = 'utf-8'  # Ensures proper encoding for suppressed output
        os.environ['ULTRALYTICS_SILENT'] = 'True'  # Specifically for ultralytics
        
        try:
            # Load model with best device selection
            model = YOLO(config_env.V8_PATH)
            """ 
            if self.use_gpu:
                # Optimize for GPU inference
                model.to('cuda')
                if hasattr(model, 'model'):
                    # Only apply half precision if GPU supports it
                    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
                        model.model.half()
            """
            # Optimize for CPU inference
            model.export(
                format="onnx",
                optimize=True,
                dynamic=True,
                batch=1,
                half=True,
                device="cpu",
                nms=True,
                opset=13
            )
            # Warmup the model with a dummy inference
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            print("Warming up model")
            model(dummy_input, verbose=False)
        
            return model

        except Exception as e:
            print(f"Error initializing model: {e}")
            return None

    def _initialize_zones(self):
        """Initialize detection zones once"""

        try:
            polygons = load_zones_config(config_env.POLYGONS)
            return [
                sv.PolygonZone(
                    polygon=polygon,
                    triggering_anchors=(sv.Position.CENTER,)
                )
                for polygon in polygons
            ]
        except Exception as e:
            print(f"Error initializing zones: {e}")
            return []

    def detect_single_frame(self):
        try:
            # Get frame from queue with timeout
            self.frame = self.preprocessed_queue.get(timeout=0.001)
        
        except queue.Empty:
            time.sleep(0.01)
            print("Waiting for preprocessed frames... bottleneck!")
            return

        except Exception as e:
            print(f"Error in detection frame extraction from preprocessing: {e}")
            time.sleep(0.01)
            return
        
        # Run detection on this frame
        try:
            # First check if frame is valid
            if self.frame is None or not isinstance(self.frame, np.ndarray) or self.frame.size == 0:
                print("ERROR: Invalid frame for detection")
                return
                
            # Check model initialization
            if self.model is None:
                print("Model is not initialized")
                self.model = self._initialize_model()
                if self.model is None:
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
                print(f"""
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
            print(f"Outer detection loop error: {e}")
            print(traceback.format_exc())
            time.sleep(0.1)
            return

        try:
            # Convert YOLO results to Supervision detections
            self.detections = sv.Detections.from_ultralytics(self.results)
        except Exception as e:
            print("ERROR in Supervision conversion!")
            import traceback
            print(f"Error creating detections: {e}")
            print(traceback.format_exc())
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
                if len(self.detections) > 0:
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
            if len(self.detections) > 0:
                processed_frame = self.box_annotator.annotate(
                    scene=processed_frame, 
                    detections=self.detections
                )
            
            # Log FPS periodically using the class frame counter
            self.frame_count += 1
            current_time = time.time()
            show_time = current_time - self.last_log_time
            if show_time >= 4.0:  # Log every 4 seconds
                elapsed = current_time - self.last_log_time
                fps = self.frame_count / elapsed
                print(f"Inference at {fps:.2f} FPS")
                self.frame_count = 0
                self.last_log_time = current_time

            if self.detections:
                self.message_active = True
#                self.message_system()
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
                    print(f"Memory cleaned after {self.frame_count} frames")

            except queue.Full:
                print("Detection queue is full, dropping frame")
                time.sleep(0.01)

            except Exception as e:
                print(f"Error: Failed to send processed frame to queue: {e}")
                return

    def alert_system(self, processed_frame):
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
        if self.message_active:
            current_time = time.time()

            if not self.message_cooldown:
                threading.Thread(target=send_twilio_message, daemon=True).start()
                print("Alert message sent")

                self.message_cooldown = True
                self.message_cooldown_start = current_time
            
            # Check if cooldown period is over
            elif current_time - self.message_cooldown_start > self.message_interval:
                self.message_cooldown = False  # Allow next message
