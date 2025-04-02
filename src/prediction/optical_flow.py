"""
optical_flow.py

Enhances YOLO detection with optical flow for improved accuracy, with optimized performance.

Author: fw7th
Date: 2025-04-02
"""

import cv2 as cv
import numpy as np
import threading
import time

class OpticalFlowDetector:
    """
    Enhanced optical flow detection with better CPU performance.
    """
    
    def __init__(self, source, detector_class=None):
        """
        Initialize the optical flow enhanced detector.
        
        Parameters
        ----------
        source : str or int
            Video source path or camera index
        detector_class : class, optional
            Custom detector class if not using the default
        """
        # Import the detector class if provided
        if detector_class:
            self.detector = detector_class(source)
        else:
            from src.prediction.detection import ObjectDetector
            self.detector = ObjectDetector(source)
            
        # Optical flow parameters
        self.prev_gray = None
        self.flow = None
        self.motion_threshold = 20  # Lower threshold for better sensitivity
        self.flow_skip_frames = 2   # Process optical flow every N frames
        self.frame_count = 0
        
        # Thread control
        self.running = False
        self.lock = threading.Lock()
        
        # Output data
        self.latest_detections = None
        self.latest_annotated = None
        self.latest_flow_map = None
        
        # Control flags
        self.initialized = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
    def initialize(self):
        """Initialize the detector and capture object"""
        try:
            self.cap, self.detector = self.detector.threaded_capture()
            if self.cap is None:
                print("OpticalFlow: Failed to initialize detector")
                return False
                
            # Start processing thread
            self.running = True
            self.thread = threading.Thread(target=self._process_loop, daemon=True)
            self.thread.start()
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"OpticalFlow initialization error: {e}")
            return False
    
    def _process_loop(self):
        """Process frames continuously with optical flow enhancements"""
        # Import the annotator
        from src.utils.general import CORNER_ANNOTATOR
        
        # Initialize variables
        self.fps_counter = 0
        last_update_time = time.time()
        
        while self.running:
            try:
                # Get latest results from detector
                detections, annotated = self.detector.get_latest_results()
                
                if annotated is None:
                    time.sleep(0.05)
                    continue
                
                # Update FPS counter
                self.fps_counter += 1
                current_time = time.time()
                if current_time - last_update_time > 5.0:
                    self.current_fps = self.fps_counter / (current_time - last_update_time)
                    print(f"OpticalFlow processing rate: {self.current_fps:.1f} FPS")
                    self.fps_counter = 0
                    last_update_time = current_time
                
                # Process optical flow only on some frames
                self.frame_count += 1
                process_flow = (self.frame_count % self.flow_skip_frames == 0)
                
                # Add FPS to the frame
                h, w = annotated.shape[:2]
                cv.putText(
                    annotated,
                    f"OF: {self.current_fps:.1f}",
                    (w-150, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
                )
                
                # Skip optical flow processing if not needed
                if not process_flow:
                    with self.lock:
                        self.latest_detections = detections
                        self.latest_annotated = annotated
                        self.latest_flow_map = None
                    continue
                
                # Convert current frame to grayscale for optical flow - downsample for speed
                downscaled = cv.resize(annotated, (annotated.shape[1]//2, annotated.shape[0]//2))
                current_gray = cv.cvtColor(downscaled, cv.COLOR_BGR2GRAY)
                
                # Initialize optical flow components if first frame
                if self.prev_gray is None:
                    self.prev_gray = current_gray
                    flow_map = None
                    
                    with self.lock:
                        self.latest_detections = detections
                        self.latest_annotated = annotated
                        self.latest_flow_map = None
                    continue
                
                # Calculate optical flow with optimized parameters
                try:
                    # Use Farneback with parameters tuned for CPU performance
                    self.flow = cv.calcOpticalFlowFarneback(
                        self.prev_gray, current_gray, 
                        None, 0.4, 2, 10, 2, 5, 1.1, 0  # Reduced parameters
                    )
                    
                    # Convert flow to polar coordinates
                    magnitude, angle = cv.cartToPolar(self.flow[..., 0], self.flow[..., 1])
                    
                    # Create motion mask based on magnitude threshold
                    motion_mask = magnitude > self.motion_threshold
                    
                    # Create flow visualization (downsampled)
                    hsv = np.zeros_like(downscaled)
                    hsv[..., 1] = 255
                    hsv[..., 0] = angle * 180 / np.pi / 2
                    hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
                    flow_map_small = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                    
                    # Resize flow map to match original frame
                    flow_map = cv.resize(flow_map_small, (annotated.shape[1], annotated.shape[0]))
                    
                    # Enhance detections with motion information
                    if detections is not None and len(detections) > 0:
                        enhanced_detections = []
                        confidence_boost = []
                        
                        # Resize motion mask to match original frame size
                        motion_mask_full = cv.resize(
                            motion_mask.astype(np.uint8) * 255, 
                            (annotated.shape[1], annotated.shape[0])
                        ) > 127
                        
                        # Enhance each detection with motion information
                        for i, box in enumerate(detections.xyxy):
                            x1, y1, x2, y2 = map(int, box)
                            # Scale coordinates for downsampled motion mask
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(annotated.shape[1], x2), min(annotated.shape[0], y2)
                            
                            # Calculate motion percentage in detection box
                            box_area = (x2 - x1) * (y2 - y1)
                            if box_area > 0:
                                # Extract motion mask for this region
                                try:
                                    motion_pixels = np.sum(motion_mask_full[y1:y2, x1:x2])
                                    motion_percentage = motion_pixels / box_area
                                    
                                    # Boost confidence based on motion
                                    if motion_percentage > 0.05:  # Lower threshold (5%)
                                        enhanced_detections.append(i)
                                        # More conservative boost (max 0.1)
                                        boost = min(0.1, motion_percentage * 0.3)
                                        confidence_boost.append(min(0.95, detections.confidence[i] + boost))
                                    else:
                                        enhanced_detections.append(i)
                                        confidence_boost.append(detections.confidence[i])
                                except Exception as e:
                                    # Fall back to original confidence on error
                                    enhanced_detections.append(i)
                                    confidence_boost.append(detections.confidence[i])
                            else:
                                # Invalid box, keep original detection
                                enhanced_detections.append(i)
                                confidence_boost.append(detections.confidence[i])
                        
                        # Create enhanced detections
                        if enhanced_detections:
                            enhanced_detections = detections[enhanced_detections]
                            enhanced_detections.confidence = np.array(confidence_boost)
                            detections = enhanced_detections
                            
                            # Re-annotate with enhanced detections
                            annotated = CORNER_ANNOTATOR.annotate(annotated, detections=detections)
                    
                    # Update previous frame
                    self.prev_gray = current_gray
                    
                    # Store results atomically
                    with self.lock:
                        self.latest_detections = detections
                        self.latest_annotated = annotated
                        self.latest_flow_map = flow_map
                        
                except Exception as e:
                    print(f"OpticalFlow error: {e}")
                    
                    # Use original detections on error
                    with self.lock:
                        self.latest_detections = detections
                        self.latest_annotated = annotated
                        self.latest_flow_map = None
                
            except Exception as e:
                print(f"Error in optical flow processing: {e}")
                time.sleep(0.1)
                
    def process_frame(self):
        """
        Process a frame with both YOLO and optical flow.
        Returns the latest processed results.
        """
        with self.lock:
            return self.latest_detections, self.latest_annotated, self.latest_flow_map

    def stop(self):
        """Stop the detector and release resources."""
        self.running = False
        
        # Wait for thread to finish
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
            
        if hasattr(self, 'detector'):
            self.detector.stop()
