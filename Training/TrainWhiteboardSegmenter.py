from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

if __name__ == '__main__':
    model.train(data="data.yaml", epochs=800, imgsz=257, device=0)