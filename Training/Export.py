from ultralytics import YOLO

path = ""
model = YOLO(path)

if __name__ == '__main__':
    model.val()