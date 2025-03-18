from src.utils.general import VideoProcessor

source = "/home/fw7th/Videos/people-walking.mp4"
save = "/home/fw7th/Videos/output.mp4"


pro = VideoProcessor(source, save)
pro.display_video()

