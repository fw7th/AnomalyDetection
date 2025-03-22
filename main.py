from src.utils.general import VideoProcessor

source = "/home/fw7th/Videos/people-walking.mp4"
RTSP_STREAM = "rtsp://localhost:8554/live0.stream"
# source = 0
save = "/home/fw7th/Videos/output.mp4"


pro = VideoProcessor(source, save)
pro.display_video()
pro.save_video()

