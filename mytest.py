from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")
print(model.task)