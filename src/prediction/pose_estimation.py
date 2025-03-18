from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

results = model("/home/fw7th/Videos/people-walking.mp4")
for result in results:
    xy = results.keypoints.xy
    kpts = results.keypoints.data
