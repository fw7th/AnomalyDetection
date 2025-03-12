import supervision as sv
from prediction.tracking import Tracker
from utils.time import ClockBasedTimer
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

class Loitering:
    def __init__(self):
        self.tracker = Tracker()

    def time_measure(self):
        pass


