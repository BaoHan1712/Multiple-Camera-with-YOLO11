from ultralytics import YOLO

model = YOLO("model/yolo11n.pt")
model.export(format="onnx", imgsz=320, half=True, opset=11, simplify=True)

