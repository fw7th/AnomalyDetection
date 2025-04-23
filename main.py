from src.utils.compile import Compile

source = "/home/fw7th/Videos/1.mp4"
#source = "rtsp://localhost:8554/live0.stream"
#source = 0
#source = "/home/fw7th/Videos/people-walking.mp4"
save = "/home/fw7th/Videos/output.mp4"


pipe = Compile(source, enable_saving=True, save_dir=save)
pipe.run()
