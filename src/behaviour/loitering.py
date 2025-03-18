import supervision as sv
from src.prediction.tracking import Tracker
from src.utils.time import ClockBasedTimer
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

class CustomSink:
    def __init__(self):
        pass

    def on_prediction(self, result: dict, frame: VideoFrame) -> None:
        pass

class Loitering:
    def __init__(self):
        self.tracker = Tracker()
        self.timer = ClockBasedTimer()
        

    def time_measure(self):
        self.tracker.get_tracked_objects()


