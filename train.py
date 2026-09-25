from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        project="runs",
        name="tower_detection"
    )

if __name__ == "__main__":
    main()
