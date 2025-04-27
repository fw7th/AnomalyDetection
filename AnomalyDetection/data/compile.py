"""
compile.py

Provides a real-time video processing pipeline with multithreaded/multiprocess architecture.
This module implements a pipeline that coordinates frame acquisition, preprocessing,
object detection, and display with optimized resource utilization and synchronization.

Author: Based on original by fw7th
Date: 2025-04-26
"""

from data import FrameProcessor_, Frames
from services import ObjectDetector, VideoDisplay
from core import LOG
from threading import Thread, Event
from multiprocessing import Process, Manager, Queue
import queue
import torch.cuda
import time
import contextlib

# Configure logging

class Compile:
    """Real-time video processing pipeline that coordinates multiple processing stages.
    
    This class implements a pipeline that uses either CPU (multiprocessing) or GPU (threading) 
    processing to handle video frame acquisition, preprocessing, object detection, and display.
    
    Attributes:
        use_gpu (bool): Whether GPU processing is available.
        source (str or int): Path to video file or camera index.
        save_dir (str): Directory to save processed frames.
        enable_saving (bool): Whether to save processed frames.
        your_num (str): Phone number for alert notifications.
        your_mail (str): Email address for alert notifications.
    """
    def __init__(self, source, enable_saving=False, save_dir=None, your_num=None, your_mail=None, accuracy=None):
        """Initialize the pipeline with the video source and configuration.
        
        Args:
            source (str or int): Path to video file or camera index.
            enable_saving (bool, optional): Whether to save processed frames. Defaults to False.
            save_dir (str, optional): Directory to save processed frames. Defaults to None.
            your_num (str, optional): Phone number for alert notifications. Defaults to None.
            your_mail (str, optional): Email address for alert notifications. Defaults to None.
            accuracy (str, int, optional): Balance of speed and accuracy. Defaults to 2 or "mid".
        """
        # Configuration
        self.use_gpu = torch.cuda.is_available()
        self.source = source
        self.save_dir = save_dir
        self.enable_saving = enable_saving
        self.your_num = your_num
        self.your_mail = your_mail
        self.accuracy = accuracy
        
        LOG.info(f"Initializing pipeline: GPU available: {self.use_gpu}")
        
        # Initialize queues based on processing mode (CPU/GPU)
        self._initialize_queues()
        
        # Initialize synchronization events
        self._initialize_events()
        
        # Initialize core modules
        self._initialize_modules()
        
        # Store workers for cleanup
        self.workers = []

    def _initialize_queues(self):
        """Initialize communication queues based on available hardware.
        
        Creates queue objects for transferring data between pipeline stages with 
        appropriate size limits to control memory usage.
        """
        # For CPU (processes), use multiprocessing Queue
        
        # Set queue max sizes to control memory usage
        reader_queue_size = 10
        detector_queue_size = 40
        other_queue_size = 10
        
        # Queue for frames from camera/video to preprocessing
        self.frame_queue = Queue(maxsize=reader_queue_size)
        
        # Queue for preprocessed frames to detection
        self.preprocessed_queue = Queue(maxsize=other_queue_size)

        # Queue for detection results to display
        self.detection_queue = Queue(maxsize=detector_queue_size)
        

    def _initialize_events(self):
        """Initialize synchronization events for pipeline components.
        
        Creates events to coordinate the starting, running, and stopping of
        different pipeline stages.
        """
        # For CPU processing, use multiprocessing events
        self.manager = Manager()
        self.events = {
            # Events for component initialization (ready to process)
            "reader_ready": self.manager.Event(),
            "preprocessing_ready": self.manager.Event(),
            "detection_ready": self.manager.Event(),
            "display_ready": self.manager.Event(),
            
            # Event to signal pipeline shutdown
            "pipeline_stop": self.manager.Event()
        }

    def _initialize_modules(self):
        """Initialize the core processing modules.
        
        Creates instances of the frame reader, preprocessor, detector, and
        display classes, configuring them with the appropriate queues.
        """
        self.reader = Frames(self.frame_queue, self.source)
        
        self.preprocessing = FrameProcessor_(
            self.frame_queue,
            self.preprocessed_queue
        )
        
        self.detect = ObjectDetector(
            self.preprocessed_queue,
            self.detection_queue,
            self.your_num,
            self.your_mail,
            self.accuracy
        )
        
        self.display = VideoDisplay(
            self.detection_queue,
            enable_saving=self.enable_saving,
            save_dir=self.save_dir
        )

    def read_frame(self):
        """Read frames from source and put them in the frame queue.
        
        Initializes the frame reader and continuously reads frames until the source
        is exhausted or the pipeline is stopped.
        """
        try:
            LOG.info("Starting frame reader")
            
            # Initialize the reader and signal it's ready to process
            self.reader.running.set()
            self.events["reader_ready"].set()
            LOG.info("Frame reader ready")
            
            # Start reading frames (this will run until the source is exhausted)
            self.reader.read_frames()
            LOG.info("Frame reader completed")
            
            # Signal that all frames have been read (this should be done by read_frames)
            LOG.info("All frames read, waiting for pipeline to complete processing")
            
            # Wait for display to process the final frames before stopping
            while not self.detection_queue.empty() and not self.events["pipeline_stop"].is_set():
                time.sleep(0.1)
                
            # Now signal pipeline to stop if not already stopped
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()
            
        except Exception as e:
            LOG.error(f"Error in frame reading: {e}")
            # On error, signal pipeline to stop
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()

        finally:
            if self.detection_queue.empty:
                time.sleep(0.5)
                # Release resources outside the loop when done and end reading
                if self.reader.cap is not None:
                    self.reader.cap.release()
                    LOG.info("Video capture released")
                    self.reader.running.clear()

    def preprocess_frame(self):
        """Preprocess frames from the frame queue continuously.
        
        Waits for frame reader to be ready, then continuously processes frames
        from the frame queue until the pipeline is stopped or no more frames
        are available.
        """
        try:
            LOG.info("Initializing frame preprocessor")
            
            # Wait for reader to be ready
            while not self.events["reader_ready"].is_set() and not self.events["pipeline_stop"].is_set():
                time.sleep(0.05)
                
            if self.events["pipeline_stop"].is_set():
                LOG.info("Preprocessing initialization aborted - pipeline stopping")
                return
                
            # Initialize preprocessor and signal it's ready
            self.preprocessing._running.set()
            self.events["preprocessing_ready"].set()
            LOG.info("Frame preprocessor ready")
            
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
                            LOG.info("No more frames to preprocess, preprocessor exiting")
                            break
                    
                    # Process available frames using appropriate method
                    """
                    if self.use_gpu:
                        self.preprocessing.process_single_frame_gpu()
                    """
                    self.preprocessing.process_frame()
                        
                except queue.Empty:
                    # Handle empty queue
                    time.sleep(0.01)
                except Exception as e:
                    LOG.error(f"Error processing frame: {e}")
                    if not self.events["pipeline_stop"].is_set():
                        self.events["pipeline_stop"].set()
                    break
            
            LOG.info("Frame preprocessor shutting down")
            
        except Exception as e:
            LOG.error(f"Error in preprocessing: {e}")
            # On error, signal pipeline to stop
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()

        finally:
            if self.reader.running == False:
                self.preprocessing._running.clear()

    def detect_on_frame(self):
        """Detect objects in preprocessed frames continuously.
        
        Waits for preprocessor to be ready, then continuously processes preprocessed
        frames for object detection until the pipeline is stopped or no more frames
        are available.
        """
        try:
            LOG.info("Initializing object detector")
            
            # Wait for preprocessor to be ready
            while not self.events["preprocessing_ready"].is_set() and not self.events["pipeline_stop"].is_set():
                time.sleep(0.05)
                
            if self.events["pipeline_stop"].is_set():
                LOG.info("Detection initialization aborted - pipeline stopping")
                return
                
            # Initialize detector and signal it's ready
            self.detect._running.set()
            self.events["detection_ready"].set()
            LOG.info("Object detector ready")
            
            # Start detecting objects continuously
            while not self.events["pipeline_stop"].is_set() and self.detect._running.is_set():
                try:
                    # Check if input queue has frames
                    if self.preprocessed_queue.empty():
                        # If preprocessor is still running or input not empty, wait for more frames
                        if hasattr(self.preprocessing, '_running') and self.preprocessing._running.is_set():
                            time.sleep(0.01)
                            continue
                        else:
                            # If preprocessor is done and queue is empty, exit
                            LOG.info("No more preprocessed frames to detect, detector exiting")
                            break
                    
                    # Process a single frame for detection
                    self.detect.detect_single_frame()
                    
                except queue.Empty:
                    # Handle empty queue (should be covered by the check above)
                    time.sleep(0.01)
                except Exception as e:
                    LOG.error(f"Error detecting objects in frame: {e}")
                    if not self.events["pipeline_stop"].is_set():
                        self.events["pipeline_stop"].set()
                    break
            
            LOG.info("Object detector shutting down")
        
        except Exception as e:
            LOG.error(f"Error in detection: {e}")
            # On error, signal pipeline to stop
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()

        finally:
            self.detect._running.clear()

    def display_frames(self):
        """Display detected objects in frames continuously.
        
        Waits for detector to be ready, then continuously displays frames with
        detection results until the pipeline is stopped or no more frames are
        available. Also monitors for user exit requests.
        """
        try:
            LOG.info("Initializing display")
            
            # Wait for detector to be ready
            while not self.events["detection_ready"].is_set() and not self.events["pipeline_stop"].is_set():
                time.sleep(0.05)
                
            if self.events["pipeline_stop"].is_set():
                LOG.info("Display initialization aborted - pipeline stopping")
                return
                
            # Initialize display and signal it's ready
            self.display.running.set()
            self.events["display_ready"].set()
            LOG.info("Display ready")
            
            # Start displaying frames continuously
            while not self.events["pipeline_stop"].is_set() and self.display.running.is_set():
                try:
                    # Check if input queue has inference frames
                    if self.detection_queue.empty():
                        # If detector is still running or input not empty, wait for more frames
                        if hasattr(self.detect, '_running') and self.detect._running.is_set():
                            time.sleep(0.01)
                            continue
                        else:
                            # If object detector is done and queue is empty, exit
                            LOG.info("No more detection frames to display, display exiting")
                            break
                    
                    # Display a single frame
                    self.display.display_video()
                    
                    # Check for user exit (e.g., pressing 'q')
                    if hasattr(self.display, 'should_exit') and self.display.should_exit():
                        LOG.info("User requested exit")
                        if not self.events["pipeline_stop"].is_set():
                            self.events["pipeline_stop"].set()
                        break
                    
                except queue.Empty:
                    # Handle empty queue (should be covered by the check above)
                    time.sleep(0.01)
                except Exception as e:
                    LOG.error(f"Error displaying frame: {e}")
                    if not self.events["pipeline_stop"].is_set():
                        self.events["pipeline_stop"].set()
                    break
            
            LOG.info("Display shutting down")
            
        except Exception as e:
            LOG.error(f"Error in display: {e}")
            # On error, signal pipeline to stop
            if not self.events["pipeline_stop"].is_set():
                self.events["pipeline_stop"].set()

        finally:
            self.display.cleanup()

    def setup_workers(self):
        """Create worker threads or processes based on available hardware.
        
        Creates thread for frame reading and display, and processes for 
        compute-intensive tasks like preprocessing and detection.
        """
        # Frame reader is always a thread
        self.worker_reader = Thread(target=self.read_frame, name="FrameReader")
        self.workers.append(self.worker_reader)

        LOG.info("Started reader thread")
        
        # Choose worker type based on GPU availability
        # For CPU, use processes for compute-intensive tasks
        self.worker_preprocess = [Process(target=self.preprocess_frame, name=f"Preprocessor-{i}")
                                  for i in range(2)]
        self.worker_detect = Process(target=self.detect_on_frame, name="Detector")
        
        self.workers.extend(self.worker_preprocess)
        self.workers.extend([self.worker_detect])

        LOG.info("Started preprocessing and detection threads")

        # Display is always a thread
        self.worker_display = Thread(target=self.display_frames, name="Display")
        self.workers.append(self.worker_display)

    def start_workers(self):
        """Start all worker threads and processes.
        
        Starts all worker threads and processes and waits for pipeline completion
        or interruption. When pipeline stops, it initiates cleanup.
        """
        LOG.info("Starting all pipeline workers")
        for worker in self.workers:
            worker.daemon = True  # Allow program to exit even if workers are running
            worker.start()
            LOG.info(f"Started {worker.name}")
            
        # Wait for pipeline to complete or be stopped
        try:
            while not self.events["pipeline_stop"].is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            LOG.info("Pipeline interrupted by user")
            self.events["pipeline_stop"].set()
            
        # Once pipeline_stop is set, start cleanup
        self.stop()

    def stop(self):
        """Stop all components of the pipeline and clean up resources.
        
        Signals all components to stop, joins worker threads and processes,
        and cleans up resources.
        """
        LOG.info("Stopping pipeline")
        
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
        
        # Stop display
        if hasattr(self.display, 'running'):
            self.display.running.clear()
        
        LOG.info("Joining workers")
        
        # Join all workers with timeout
        for worker in self.workers:
            try:
                worker.join(timeout=2.0)
                if worker.is_alive():
                    LOG.warning(f"Worker {worker.name} did not terminate properly")
            except Exception as e:
                LOG.error(f"Error joining {worker.name}: {e}")
        
        # Clear all events
        for event_name, event in self.events.items():
            if event_name != "pipeline_stop":  # Keep pipeline_stop set
                event.clear()
        
        # Clean up queues
        self._clean_queues()
        
        LOG.info("Pipeline stopped")

    def _clean_queues(self):
        """Clean up queues by removing any remaining items.
        
        Empties all queues to prevent resource leaks during cleanup.
        """
        queues = [
            self.frame_queue, 
            self.preprocessed_queue, 
            self.detection_queue,
        ]
        
        for queue in queues:
            try:
                # Empty the queue
                with contextlib.suppress(Exception):
                    while not queue.empty():
                        queue.get_nowait()
            except Exception as e:
                LOG.error(f"Error cleaning queue: {e}")

    def run(self):
        """Run the complete pipeline, from setup to completion.
        
        This is the main entry point for pipeline execution. Sets up workers,
        starts them, and handles any exceptions during execution.
        
        Returns:
            None
            
        Raises:
            Any exceptions that occur during pipeline execution that aren't
            handled internally.
        """
        try:
            LOG.info("Starting pipeline execution")
            self.setup_workers()
            self.start_workers()
            LOG.info("Pipeline execution completed")
        except KeyboardInterrupt:
            LOG.info("Pipeline interrupted by user")
            self.stop()
        except Exception as e:
            LOG.error(f"Pipeline error: {e}")
            self.stop()
            raise
        finally:
            self._clean_queues()
