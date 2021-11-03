from yolov5 import train

train.run(imgsz=256, batch_size=16, epochs=3, weights='yolov5s.pt', data='data.yaml')
