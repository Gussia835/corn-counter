from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

model.train(
    data='corn.yaml',
    epochs=100,
    imgsz=320,
    batch=2,
    name='corn-seg'
)
