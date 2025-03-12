from utils.general import VideoProcessor

source = "/home/fw7th/Videos/people-walking.mp4"
save = "/home/fw7th/Videos/output.mp4"


pro = VideoProcessor(source, save)
pro.process_track()


"""
detector = ObjectDetector()
detector.detect_objects(source)
"""



