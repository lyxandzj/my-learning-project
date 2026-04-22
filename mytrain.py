from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo11n.pt")
    model.train(
        data=r"VisDrone.yaml",
        epochs=3,
        imgsz=1280,
        # 批次
        batch=-1,
        cache="ram",
        workers=0,
    )