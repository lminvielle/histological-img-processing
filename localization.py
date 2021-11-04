import cv2
import glob
from yolov5 import train, YOLOv5

# train.run(imgsz=256, batch_size=16, epochs=300, weights='yolov5s.pt', data='data.yaml')

path_weights = "runs/train/exp15/weights/best.pt"
path_imgs = "../data/localization/test/"

files_test = glob.glob(path_imgs + '*.png')

yolo = YOLOv5(path_weights)
results = yolo.predict(files_test)
# predictions = results.pred[0]
for im, predictions in zip(results.imgs, results.pred):
    predictions = predictions.detach().to('cpu').numpy()

    im_pred = im.copy()
    for i in range(predictions.shape[0]):
        print(predictions[i])
        im_pred = cv2.rectangle(im_pred, tuple(predictions[i, [0, 2]].astype(int)), tuple(predictions[i, [1, 3]].astype(int)), (0, 255, 200), 1)
    cv2.imshow('image', im)
    cv2.imshow('image pred', im_pred)
    cv2.waitKey(0)
    cv2.destroyAllWindows
