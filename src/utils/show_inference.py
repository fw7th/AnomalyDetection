import cv2 as cv
from src.prediction.tracking import Tracker
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from src import config


RTSP_STREAM = "rtsp://localhost:8554/live0.stream"
model_name = config.MODEL_PATH
track = Tracker()

pipeline = InferencePipeline.init(
    model_id= model_name,
    video_reference=RTSP_STREAM,
    on_prediction=track.get_tracked_objects()
)

pipeline.start()
pipeline.join()
