from src.utils.compile import Compile

source = "/home/fw7th/Videos/1.mp4"
#source = "rtsp://localhost:8554/live0.stream"
#source = 0
save = "/home/fw7th/Videos/output.mp4"


pipe = Compile(source)
pipe.setup_and_start()
pipe.start()
pipe.stop()
