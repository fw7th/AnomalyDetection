import cv2 as cv
import numpy as np
import threading
import time
import queue
import supervision as sv
from src.prediction.optical_flow import OpticalFlowDetector
from src.prediction.tracking import ObjectTracker

class VideoProcessor:
    """
    Optimized video processing class that handles frame capture, processing,
    and display with efficient multithreading and resource management.
    """
    def __init__(self, source_path, save_dir: str, buffer_size: int = 100):
        """
        Initialize the video processor with source and output paths.
        
        Parameters:
        -----------
        source_path : str
            Path to the source video or camera index
        save_dir : str
            Directory to save processed video output
        buffer_size : int, optional
            Size of the frame buffer queue (default: 100)
        """
        self.source_path = source_path
        self.save_dir = save_dir
        
        # Initialize components with proper error handling
        self.flow_detector = OpticalFlowDetector(self.source_path)
        self.tracker = ObjectTracker(input_queue_size=20)  # Increased queue size
        
        # Thread-safe queue for frame buffering
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        
        # Event to signal thread termination
        self.stop_event = threading.Event()
        
        # Performance monitoring
        self.stats = {
            "display_fps": 0,
            "last_fps_update": time.time(),
            "frames_processed": 0,
            "queue_high_water_mark": 0
        }
        
        # Store frames for saving (optional)
        self.save_frames = []
        self.save_frames_lock = threading.Lock()
        self.max_save_frames = 1000  # Limit to prevent memory issues

    def display_video(self, window_name="Tracking", enable_saving=False):
        """
        Display the processed video with optimized performance and provide
        options for saving frames.
        
        Parameters:
        -----------
        window_name : str, optional
            Name of the display window (default: "Tracking")
        enable_saving : bool, optional
            Whether to store frames for later saving (default: False)
        """
        # Create optimized window
        cv.namedWindow(window_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(window_name, 1280, 720)  # Reasonable default size
        
        # Start tracking thread first to ensure it's ready
        tracking_success = self.tracker.start_tracking()
        if not tracking_success:
            print("Error: Failed to start tracker")
            return
        
        # Initialize flow detector
        detector_success = self.flow_detector.initialize()
        if not detector_success:
            print("Error: Failed to initialize flow detector")
            self.tracker.stop()
            return
            
        # Start processing thread
        processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        processing_thread.start()
        
        try:
            display_start_time = time.time()
            frames_displayed = 0
            empty_frame_count = 0
            
            # Main display loop
            while not self.stop_event.is_set():
                try:
                    # Non-blocking frame retrieval with timeout
                    labeled_frame = self.frame_queue.get(timeout=0.5)
                    
                    # Update statistics
                    frames_displayed += 1
                    current_time = time.time()
                    if current_time - self.stats["last_fps_update"] >= 3.0:
                        elapsed = current_time - self.stats["last_fps_update"]
                        self.stats["display_fps"] = frames_displayed / elapsed
                        self.stats["queue_high_water_mark"] = max(
                            self.stats["queue_high_water_mark"], 
                            self.frame_queue.qsize()
                        )
                        
                        # Log performance stats
                        print(f"Display fps: {self.stats['display_fps']:.1f}, "
                              f"Queue: {self.frame_queue.qsize()}/{self.frame_queue.maxsize}")
                        
                        # Reset counters
                        frames_displayed = 0
                        self.stats["last_fps_update"] = current_time
                    
                    # Handle frame display
                    if labeled_frame is not None and labeled_frame.size > 0:
                        # Add performance stats to frame
                        h, w = labeled_frame.shape[:2]
                        cv.putText(
                            labeled_frame,
                            f"Display FPS: {self.stats['display_fps']:.1f}",
                            (10, h - 10), 
                            cv.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 255), 1
                        )
                        
                        # Display the frame
                        cv.imshow(window_name, labeled_frame)
                        empty_frame_count = 0
                        
                        # Save frame if enabled
                        if enable_saving:
                            with self.save_frames_lock:
                                if len(self.save_frames) < self.max_save_frames:
                                    self.save_frames.append(labeled_frame.copy())
                    else:
                        # Handle empty frames
                        empty_frame_count += 1
                        if empty_frame_count > 10:
                            # Display waiting message
                            waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv.putText(
                                waiting_frame,
                                "Waiting for frames...",
                                (50, 240),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                            )
                            cv.imshow(window_name, waiting_frame)
                    
                    # Check for exit key with minimal blocking
                    key = cv.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and enable_saving:
                        # Trigger immediate save on 's' key
                        print("Saving video...")
                        save_thread = threading.Thread(target=self.save_video)
                        save_thread.daemon = True
                        save_thread.start()
                
                except queue.Empty:
                    # Display waiting message
                    waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv.putText(
                        waiting_frame,
                        "Waiting for frames...",
                        (50, 240),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                    )
                    cv.imshow(window_name, waiting_frame)
                    
                    # Check if tracker or detector has died
                    if self.tracker.get_error_state():
                        print("Tracker encountered an error. Stopping.")
                        break
        
        finally:
            # Clean shutdown
            print("Shutting down display...")
            self.stop_event.set()
            
            # Stop components in correct order
            self.flow_detector.stop()
            self.tracker.stop()
            
            # Close windows
            cv.destroyAllWindows()

    def _process_frames(self):
        """
        Dedicated thread for processing frames through the detection and tracking pipeline.
        """
        try:
            print("Starting frame processing worker...")
            frame_count = 0
            last_fps_time = time.time()
            
            while not self.stop_event.is_set():
                # Get detections and annotated frame from the flow detector
                detections, annotated_frame, flow_map = self.flow_detector.process_frame()
                
                # Performance tracking
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_time > 5.0:
                    fps = frame_count / (current_time - last_fps_time)
                    print(f"Processing pipeline: {fps:.1f} FPS")
                    frame_count = 0
                    last_fps_time = current_time
                
                # Skip if no frame available
                if annotated_frame is None:
                    time.sleep(0.01)  # Small delay to prevent CPU hogging
                    continue
                    
                # Add detections to tracker if available
                if detections is not None:
                    self.tracker.add_detection_for_tracking(detections, annotated_frame)
                
                # Get the labeled frame with tracking information
                labeled_frame = self.tracker.return_frames()
                
                # If tracking hasn't produced a frame yet, use the annotated frame
                if labeled_frame is None:
                    labeled_frame = annotated_frame
                    
                # Add flow visualization if available
                if flow_map is not None and labeled_frame is not None:
                    # Ensure compatible shapes
                    if flow_map.shape[:2] != labeled_frame.shape[:2]:
                        try:
                            flow_map = cv.resize(flow_map, (labeled_frame.shape[1], labeled_frame.shape[0]))
                        except Exception as e:
                            print(f"Flow map resize error: {e}")
                            flow_map = None
                    
                    # Overlay flow map with transparency
                    if flow_map is not None:
                        try:
                            # Use addWeighted for overlay with proper alpha blending
                            labeled_frame = cv.addWeighted(labeled_frame, 0.7, flow_map, 0.3, 0)
                        except Exception as e:
                            print(f"Flow map overlay error: {e}")
                
                # Queue the frame with timeout to prevent blocking
                try:
                    if labeled_frame is not None:
                        self.frame_queue.put(labeled_frame.copy(), timeout=0.1)
                except queue.Full:
                    # If queue is full, remove oldest frame to make room
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(labeled_frame.copy(), timeout=0.1)
                    except (queue.Empty, queue.Full):
                        pass  # Skip frame if queue management fails
                
                # Brief sleep to prevent tight loops
                time.sleep(0.005)
                
        except Exception as e:
            print(f"Processing thread error: {e}")
            import traceback
            traceback.print_exc()
            
            # Signal to stop the application
            self.stop_event.set()

    def save_video(self, output_path=None):
        """
        Save the processed video to disk.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the video file (default: uses self.save_dir)
        """
        save_path = output_path if output_path else self.save_dir
        
        with self.save_frames_lock:
            frames = self.save_frames.copy()
            # Clear the frames list to free memory
            self.save_frames = []
        
        if not frames:
            print("No frames to save.")
            return
        
        try:
            # Get video information for proper output settings
            vid_info = sv.VideoInfo.from_video_path(self.source_path)
            
            # Ensure all frames are the same size
            first_frame = frames[0]
            frame_size = (first_frame.shape[1], first_frame.shape[0])
            
            # Create video writer with proper codec
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(
                save_path, 
                fourcc, 
                vid_info.fps, 
                frame_size
            )
            
            # Write frames with progress reporting
            total_frames = len(frames)
            for i, frame in enumerate(frames):
                # Ensure consistent frame size
                if frame.shape[1] != frame_size[0] or frame.shape[0] != frame_size[1]:
                    frame = cv.resize(frame, frame_size)
                
                # Write the frame
                out.write(frame)
                
                # Report progress occasionally
                if (i+1) % 100 == 0 or i+1 == total_frames:
                    print(f"Saving progress: {i+1}/{total_frames} frames ({(i+1)/total_frames*100:.1f}%)")
            
            # Release resources
            out.release()
            print(f"Video saved successfully to {save_path}")
            
        except Exception as e:
            print(f"Error saving video: {e}")
            import traceback
            traceback.print_exc()
            
    def run(self, enable_saving=True):
        """
        Convenience method to run the video processing pipeline.
        
        Parameters:
        -----------
        enable_saving : bool, optional
            Whether to enable frame saving (default: True)
        """
        try:
            self.display_video(enable_saving=enable_saving)
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        finally:
            # Ensure cleanup
            self.stop_event.set()
            if enable_saving and self.save_frames:
                print("Saving video before exit...")
                self.save_video()
