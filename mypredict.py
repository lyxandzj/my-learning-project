from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")
model.predict(
    source=0,
    save=False,
    show=True,
)