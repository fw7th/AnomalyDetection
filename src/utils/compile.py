from src.prediction.process_frame import FrameProcessor_
from src.prediction.tracking import ObjectTracker
from src.prediction.detection import ObjectDetector
from src.prediction.display import VideoDisplay
from src.prediction.frame_read import Frames
from threading import Thread, Event
from multiprocessing import Process, Manager, Queue
import queue
import torch.cuda
import logging
import time
import contextlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Pipeline")

class Compile:
    """
    Real-time video processing pipeline that coordinates multiple processing stages
    using either CPU (multiprocessing) or GPU (threading) processing.
    """
    def __init__(self, source, enable_saving=False, save_dir=None):
        """
        Initialize the pipeline with the video source and configuration.
        
        Args:
            source: Path to video or camera index
            enable_saving: Whether to save processed frames
            save_dir: Directory to save processed frames
        """
        # Configuration
        self.use_gpu = torch.cuda.is_available()
        self.source = source
        self.save_dir = save_dir
        self.enable_saving = enable_saving
        
        logger.info(f"Initializing pipeline: GPU available: {self.use_gpu}")
        
        # Initialize queues based on processing mode (CPU/GPU)
        self._initialize_queues()
        
        # Initialize synchronization events
        self._initialize_events()
        
        # Initialize core modules
        self._initialize_modules()
        
        # Store workers for cleanup
        self.workers = []

    def _initialize_queues(self):
        """Initialize communication queues based on available hardware."""
        # For GPU (threads), use thread-safe Queue
        # For CPU (processes), use multiprocessing Queue
        
        # Set queue max sizes to control memory usage
        max_queue_size = 15  # Limit memory consumption
        
        # Queue for frames from camera/video to preprocessing
        self.frame_queue = queue.Queue(maxsize=max_queue_size) if self.use_gpu else Queue(maxsize=max_queue_size)
        
        # Queue for preprocessed frames to detection
        self.preprocessed_queue = queue.Queue(maxsize=max_queue_size) if self.use_gpu else Queue(maxsize=max_queue_size)
        
        # Queue for detection results to tracking
        self.detection_queue = queue.Queue(maxsize=max_queue_size) if self.use_gpu else Queue(maxsize=max_queue_size)
        
        # Queue for tracking results to display
        self.tracker_queue = queue.Queue(maxsize=max_queue_size) if self.use_gpu else Queue(maxsize=max_queue_size)

    def _initialize_events(self):
        """Initialize synchronization events based on processing mode."""
        if self.use_gpu:
            # For GPU processing, use thread events
            self.events = {
                # Events for component initialization (ready to process)
                "reader_ready": Event(),
                "preprocessing_ready": Event(),
                "detection_ready": Event(),
                "tracking_ready": Event(),
                "display_ready": Event(),
                
                # Event to signal pipeline shutdown
                "pipeline_stop": Event()
            }
        else:
            # For CPU processing, use multiprocessing events
            self.manager = Manager()
            self.events = {
                # Events for component initialization (ready to process)
                "reader_ready": self.manager.Event(),
                "preprocessing_ready": self.manager.Event(),
                "detection_ready": self.manager.Event(),
                "tracking_ready": self.manager.Event(),
                "display_ready": self.manager.Event(),
                
                # Event to signal pipeline shutdown
                "pipeline_stop": self.manager.Event()
            }

    def _initialize_modules(self):
        """Initialize the core processing modules."""
        self.reader = Frames(self.frame_queue, self.source)
        
        self.preprocessing = FrameProcessor_(
            self.frame_queue,
            self.preprocessed_queue
        )
        
        self.detect = ObjectDetector(
            self.preprocessed_queue,
            self.detection_queue
        )
        
        self.track = ObjectTracker(
            self.detection_queue,
            self.tracker_queue
        )
        
        self.display = VideoDisplay(
            self.tracker_queue,
            enable_saving=self.enable_saving,
            save_dir=self.save_dir
        )

    def read_frame(self):
        """Read frames from source and put them in the frame queue."""
        try:
            logger.info("Starting frame reader")
            
            # Initialize the reader and signal it's ready to process
            self.reader.running.set()
            self.events["reader_ready"].set()
            logger.info("Frame reader ready")
            
            # Start reading frames (this will run until the source is exhausted)
            self.reader.read_frames()
            logger.info("Frame reader completed")
            
            # Signal that all frames have been read (this should be done by read_frames)
            logger.info("All frames read, waiting for pipeline to complete processing")
            
            # Wait for display to process the final frames before stopping
            while not self.tracker_queue.empty() and not self.events["pipeline_stop"].is_set():
                time.sleep(0.1)
                
            # Now signal pipeline to stop if not already stopped
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()
            
        except Exception as e:
            logger.error(f"Error in frame reading: {e}")
            # On error, signal pipeline to stop
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()

        finally:
            # Release resources outside the loop when done
            if self.reader.cap is not None:
                self.reader.cap.release()
                print("Video capture released")
            self.reader.running.clear()

    def preprocess_frame(self):
        """Preprocess frames from the frame queue continuously."""
        try:
            logger.info("Initializing frame preprocessor")
            
            # Wait for reader to be ready
            while not self.events["reader_ready"].is_set() and not self.events["pipeline_stop"].is_set():
                time.sleep(0.05)
                
            if self.events["pipeline_stop"].is_set():
                logger.info("Preprocessing initialization aborted - pipeline stopping")
                return
                
            # Initialize preprocessor and signal it's ready
            self.preprocessing._running.set()
            self.events["preprocessing_ready"].set()
            logger.info("Frame preprocessor ready")
            
            # Start processing frames continuously
            while not self.events["pipeline_stop"].is_set() and self.preprocessing._running.is_set():
                try:
                    # Check if input queue has frames
                    if self.frame_queue.empty():
                        # If reader is still running or queue not empty, wait for more frames
                        if hasattr(self.reader, 'running') and self.reader.running.is_set():
                            time.sleep(0.01)
                            continue
                        else:
                            # If reader is done and queue is empty, exit
                            logger.info("No more frames to preprocess, preprocessor exiting")
                            break
                    
                    # Process available frames using appropriate method
                    if self.use_gpu:
                        self.preprocessing.process_single_frame_gpu()
                    else:
                        self.preprocessing.process_single_frame_cpu()
                        
                except queue.Empty:
                    # Handle empty queue
                    time.sleep(0.01)
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    if not self.events["pipeline_stop"].is_set():
                        self.events["pipeline_stop"].set()
                    break
            
            logger.info("Frame preprocessor shutting down")
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            # On error, signal pipeline to stop
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()

        finally:
            self.preprocessing._running.clear()

    def detect_on_frame(self):
        """Detect objects in preprocessed frames continuously."""
        try:
            logger.info("Initializing object detector")
            
            # Wait for preprocessor to be ready
            while not self.events["preprocessing_ready"].is_set() and not self.events["pipeline_stop"].is_set():
                time.sleep(0.05)
                
            if self.events["pipeline_stop"].is_set():
                logger.info("Detection initialization aborted - pipeline stopping")
                return
                
            # Initialize detector and signal it's ready
            self.detect._running.set()
            self.events["detection_ready"].set()
            logger.info("Object detector ready")
            
            # Start detecting objects continuously
            while not self.events["pipeline_stop"].is_set():
                try:
                    # Check if input queue has frames
                    if self.preprocessed_queue.empty():
                        # If preprocessor is still running or input not empty, wait for more frames
                        if hasattr(self.preprocessing, '_running') and self.preprocessing._running.is_set():
                            time.sleep(0.01)
                            continue
                        else:
                            # If preprocessor is done and queue is empty, exit
                            logger.info("No more preprocessed frames to detect, detector exiting")
                            break
                    
                    # Process a single frame for detection
                    self.detect.detect_single_frame()
                    
                except queue.Empty:
                    # Handle empty queue (should be covered by the check above)
                    time.sleep(0.01)
                except Exception as e:
                    logger.error(f"Error detecting objects in frame: {e}")
                    if not self.events["pipeline_stop"].is_set():
                        self.events["pipeline_stop"].set()
                    break
            
            logger.info("Object detector shutting down")
        
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            # On error, signal pipeline to stop
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()

        finally:
            self.detect._running.clear()

    def track_detections(self):
        """Track detected objects across frames continuously."""
        try:
            logger.info("Initializing object tracker")
            
            # Wait for detector to be ready
            while not self.events["detection_ready"].is_set() and not self.events["pipeline_stop"].is_set():
                time.sleep(0.05)
                
            if self.events["pipeline_stop"].is_set():
                logger.info("Tracking initialization aborted - pipeline stopping")
                return
                
            # Initialize tracker and signal it's ready
            self.track._running.set()
            self.events["tracking_ready"].set()
            logger.info("Object tracker ready")
            
            # Start tracking objects continuously
            while not self.events["pipeline_stop"].is_set():
                try:
                    # Check if input queue has detection results
                    if self.detection_queue.empty():
                        # If detector is still running or input not empty, wait for more results
                        if hasattr(self.detect, '_running') and self.detect._running.is_set():
                            time.sleep(0.01)
                            continue
                        else:
                            # If detector is done and queue is empty, exit
                            logger.info("No more detection results to track, tracker exiting")
                            break
                    
                    # Process a single frame for tracking
                    if self.use_gpu:
                        self.track.gpu_track()
                    else:
                        self.track.cpu_track()
                    
                except queue.Empty:
                    # Handle empty queue (should be covered by the check above)
                    time.sleep(0.01)
                except Exception as e:
                    logger.error(f"Error tracking objects in frame: {e}")
                    if not self.events["pipeline_stop"].is_set():
                        self.events["pipeline_stop"].set()
                    break
            
            logger.info("Object tracker shutting down")
            
        except Exception as e:
            logger.error(f"Error in tracking: {e}")
            # On error, signal pipeline to stop
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()

        finally:
            self.track._running.clear()

    def display_frames(self):
        """Display tracked objects in frames continuously."""
        try:
            logger.info("Initializing display")
            
            # Wait for tracker to be ready
            while not self.events["tracking_ready"].is_set() and not self.events["pipeline_stop"].is_set():
                time.sleep(0.05)
                
            if self.events["pipeline_stop"].is_set():
                logger.info("Display initialization aborted - pipeline stopping")
                return
                
            # Initialize display and signal it's ready
            self.display.running.set()
            self.events["display_ready"].set()
            logger.info("Display ready")
            
            # Start displaying frames continuously
            while not self.events["pipeline_stop"].is_set():
                try:
                    # Check if input queue has tracked frames
                    if self.tracker_queue.empty():
                        # If tracker is still running or input not empty, wait for more frames
                        if hasattr(self.track, '_running') and self.track._running.is_set():
                            time.sleep(0.01)
                            continue
                        else:
                            # If tracker is done and queue is empty, exit
                            logger.info("No more tracked frames to display, display exiting")
                            break
                    
                    # Display a single frame
                    self.display.display_video()
                    
                    # Check for user exit (e.g., pressing 'q')
                    if hasattr(self.display, 'should_exit') and self.display.should_exit():
                        logger.info("User requested exit")
                        if not self.events["pipeline_stop"].is_set():
                            self.events["pipeline_stop"].set()
                        break
                    
                except queue.Empty:
                    # Handle empty queue (should be covered by the check above)
                    time.sleep(0.01)
                except Exception as e:
                    logger.error(f"Error displaying frame: {e}")
                    if not self.events["pipeline_stop"].is_set():
                        self.events["pipeline_stop"].set()
                    break
            
            logger.info("Display shutting down")
            
        except Exception as e:
            logger.error(f"Error in display: {e}")
            # On error, signal pipeline to stop
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()

        finally:
            self.display.running.clear()

    def setup_workers(self):
        """Create worker threads or processes based on available hardware."""
        # Frame reader is always a thread
        self.worker_reader = Thread(target=self.read_frame, name="FrameReader")
        self.workers.append(self.worker_reader)

        logger.info("Started reader thread")
        
        # Choose worker type based on GPU availability
        if self.use_gpu:
            # For GPU, use threads for all workers
            self.worker_preprocess = Thread(target=self.preprocess_frame, name="Preprocessor")
            self.worker_detect = Thread(target=self.detect_on_frame, name="Detector")
            self.worker_track = Thread(target=self.track_detections, name="Tracker")
        else:
            # For CPU, use processes for compute-intensive tasks
            self.worker_preprocess = Process(target=self.preprocess_frame, name="Preprocessor")
            self.worker_detect = Process(target=self.detect_on_frame, name="Detector")
            self.worker_track = Process(target=self.track_detections, name="Tracker")
        
        self.workers.extend([
            self.worker_preprocess,
            self.worker_detect,
            self.worker_track
        ])

        logger.info("Started preprocessing, detection and tracking threads")

        # Display is always a thread
        self.worker_display = Thread(target=self.display_frames, name="Display")
        self.workers.append(self.worker_display)

    def start_workers(self):
        """Start all worker threads and processes."""
        logger.info("Starting all pipeline workers")
        for worker in self.workers:
            worker.daemon = True  # Allow program to exit even if workers are running
            worker.start()
            logger.info(f"Started {worker.name}")
            
        # Wait for pipeline to complete or be stopped
        try:
            while not self.events["pipeline_stop"].is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            self.events["pipeline_stop"].set()
            
        # Once pipeline_stop is set, start cleanup
        self.stop()

    def stop(self):
        """Stop all components of the pipeline and clean up resources."""
        logger.info("Stopping pipeline")
        
        # Signal all components to stop
        if not self.events["pipeline_stop"].is_set():
            self.events["pipeline_stop"].set()
        
        # Stop all components by clearing their running events
        # Stop reader
        if hasattr(self.reader, 'running'):
            self.reader.running.clear()
        
        # Stop preprocessing
        if hasattr(self.preprocessing, '_running'):
            self.preprocessing._running.clear()
        
        # Stop detection
        if hasattr(self.detect, '_running'):
            self.detect._running.clear()
        
        # Stop tracking
        if hasattr(self.track, '_running'):
            self.track._running.clear()
        
        # Stop display
        if hasattr(self.display, 'running'):
            self.display.running.clear()
        
        logger.info("Joining workers")
        
        # Join all workers with timeout
        for worker in self.workers:
            try:
                worker.join(timeout=2.0)
                if worker.is_alive():
                    logger.warning(f"Worker {worker.name} did not terminate properly")
            except Exception as e:
                logger.error(f"Error joining {worker.name}: {e}")
        
        # Clear all events
        for event_name, event in self.events.items():
            if event_name != "pipeline_stop":  # Keep pipeline_stop set
                event.clear()
        
        # Clean up queues
        self._clean_queues()
        
        logger.info("Pipeline stopped")

    def _clean_queues(self):
        """Clean up queues by removing any remaining items."""
        queues = [
            self.frame_queue, 
            self.preprocessed_queue, 
            self.detection_queue,
            self.tracker_queue
        ]
        
        for queue in queues:
            try:
                # Empty the queue
                with contextlib.suppress(Exception):
                    while not queue.empty():
                        queue.get_nowait()
            except Exception as e:
                logger.error(f"Error cleaning queue: {e}")

    def run(self):
        """
        Run the complete pipeline, from setup to completion.
        This is the main entry point for pipeline execution.
        """
        try:
            logger.info("Starting pipeline execution")
            self.setup_workers()
            self.start_workers()
            logger.info("Pipeline execution completed")
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            self.stop()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.stop()
            raise
        finally:
            self._clean_queues()
