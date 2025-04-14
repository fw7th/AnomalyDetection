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
        
        # Queue for frames from camera/video to preprocessing
        self.frame_queue = queue.Queue()
        
        # Queue for preprocessed frames to detection
        self.preprocessed_queue = queue.Queue() if self.use_gpu else Queue()
        
        # Queue for detection results to tracking
        self.detection_queue = queue.Queue() if self.use_gpu else Queue()
        
        # Queue for tracking results to display
        self.tracker_queue = queue.Queue() if self.use_gpu else Queue()

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
                
                # Events for component completion (all data processed)
                "reader_done": Event(),
                "preprocessing_done": Event(),
                "detection_done": Event(),
                "tracking_done": Event(),
                "display_done": Event(),
                
                # Global pipeline control
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
                
                # Events for component completion (all data processed)
                "reader_done": self.manager.Event(),
                "preprocessing_done": self.manager.Event(),
                "detection_done": self.manager.Event(),
                "tracking_done": self.manager.Event(),
                "display_done": self.manager.Event(),
                
                # Global pipeline control
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
            logger.info("Frame reader processing frames")
            
            # Signal that all frames have been read
            self.events["reader_done"].set()
            
        except Exception as e:
            logger.error(f"Error in frame reading: {e}")
            # On error, signal pipeline to stop
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()

        finally:
            self.reader.running.clear()

    def preprocess_frame(self):
        """Preprocess frames from the frame queue."""
        try:
            logger.info("Initializing frame preprocessor")
            
            # Wait for reader to be ready before starting
            logger.info("Waiting for frame reader to be ready")
            while not self.events["reader_done"].is_set():
                time.sleep(0.05)
                
            if self.events["pipeline_stop"].is_set():
                logger.info("Preprocessing initialization aborted - pipeline stopping")
                return
                
            # Initialize preprocessor and signal it's ready
            self.preprocessing._running.set()
            self.events["preprocessing_ready"].set()
            logger.info("Frame preprocessor ready")
            
            # Start processing frames
            if self.use_gpu:
                self.preprocessing.preprocess_gpu()
            else:
                self.preprocessing.preprocess_cpu()
                
            logger.info("Frame preprocessor processing frames")
            
            # Signal that all preprocessing is done
            self.events["preprocessing_done"].set()
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            # On error, signal pipeline to stop
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()

        finally:
            self.preprocessing._running.clear()

    def detect_on_frame(self):
        """Detect objects in preprocessed frames."""
        try:
            logger.info("Initializing object detector")
            
            # Wait for preprocessor to be ready before starting
            logger.info("Waiting for preprocessor to be ready")
            while not self.events["preprocessing_done"].is_set():
                time.sleep(0.05)
                
            if self.events["pipeline_stop"].is_set():
                logger.info("Detection initialization aborted - pipeline stopping")
                return
                
            # Initialize detector and signal it's ready
            self.detect._running.set()
            self.events["detection_ready"].set()
            logger.info("Object detector ready")
            
            # Start detecting objects
            self.detect._detect_humans()
            
            logger.info("Object detector finished processing all frames")
            
            # Signal that all detection is done
            self.events["detection_done"].set()
        
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            # On error, signal pipeline to stop
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()

        finally:
            self.detect._running.clear()

    def track_detections(self):
        """Track detected objects across frames."""
        try:
            logger.info("Initializing object tracker")
            
            # Wait for detector to be ready before starting
            logger.info("Waiting for detector to be ready")
            while not self.events["detection_done"].is_set():
                time.sleep(0.05)
                
            if self.events["pipeline_stop"].is_set():
                logger.info("Tracking initialization aborted - pipeline stopping")
                return
                
            # Initialize tracker and signal it's ready
            self.track._running.set()
            self.events["tracking_ready"].set()
            logger.info("Object tracker ready")
            
            # Start tracking objects
            if self.use_gpu:
                self.track.gpu_track()
            else:
                self.track.cpu_track()
                
            logger.info("Object tracker finished processing all frames")
            
            # Signal that all tracking is done
            self.events["tracking_done"].set()
            
        except Exception as e:
            logger.error(f"Error in tracking: {e}")
            # On error, signal pipeline to stop
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()

        finally:
            self.track._running.clear()

    def display_frames(self):
        """Display tracked objects in frames."""
        try:
            logger.info("Initializing display")
            
            # Wait for tracker to be ready before starting
            logger.info("Waiting for tracker to be ready")
            while not self.events["tracking_done"].is_set():
                time.sleep(0.05)
                
            if self.events["pipeline_stop"].is_set():
                logger.info("Display initialization aborted - pipeline stopping")
                return
                
            # Initialize display and signal it's ready
            self.display.running.set()
            self.events["display_ready"].set()
            logger.info("Display ready")
            
            # Start displaying frames
            self.display.display_video()
            
            logger.info("Display finished processing all frames")
            
            # Signal that all display is done
            self.events["display_done"].set()
            
            # If display is done, trigger pipeline shutdown
            if not self.events["pipeline_stop"].is_set():
                logger.info("Display completed - signaling pipeline to shutdown")
                self.events["pipeline_stop"].set()
                
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

    def wait_for_completion(self):
        """Wait for the pipeline to complete or be stopped."""
        logger.info("Waiting for pipeline completion")
        
        # Wait for pipeline stop event
        self.events["pipeline_stop"].wait()
        logger.info("Pipeline stop event detected")
        
        # Allow a short grace period for components to finish
        time.sleep(0.5)
        
        # Stop all components
        self.stop()

    def stop(self):
        """Stop all components of the pipeline and clean up resources."""
        logger.info("Stopping pipeline")
        
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
            self.wait_for_completion()
            logger.info("Pipeline execution completed")
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            self.stop()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.stop()
            raise
