from ultralytics import YOLO

model = YOLO("runs/bestresult/ELDDS/all/weights/best.pt")
model.predict(source="D:\Download\ELDDS1400c5-dataset.v1i.yolov8\\val\images", save=True,  name='output')
