import multiprocessing as mp
from queue import Queue
import torch.cuda

class SharedQueue:
    def __init__(self):
        self.use_gpu = torch.cuda.is_available()

    def create_queue(self):
        """Create the appropriate queue depending on whether GPU or CPU is being used."""
        maxsize=25
        if self.use_gpu:
            # Use a normal Queue for threading (GPU tasks)
            _queue = Queue(maxsize)
        else:
            # Use a multiprocessing Queue for CPU tasks
            _queue = mp.Queue(maxsize)
        return _queue
