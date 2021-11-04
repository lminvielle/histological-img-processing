import cv2
import glob
import json
import numpy as np
from yolov5 import train, YOLOv5

TRAIN = 1
TEST = 0

if TRAIN:
    train.run(imgsz=256, batch_size=16, epochs=500, weights='yolov5m.pt', data='data.yaml')


if TEST:
    path_weights = "runs/train/exp7/weights/best.pt"
    path_imgs = "../data/localization/test/"

    files_test = glob.glob(path_imgs + '*.png')

    # predict
    yolo = YOLOv5(path_weights)
    results = yolo.predict(files_test)
    # read json file
    path_json = "../data/localization/boxes_test.json"
    f = open(path_json, "r")
    annotations = json.loads(f.read())
    f.close()

    for filename, im, predictions in zip(results.files, results.imgs, results.pred):
        print(filename)
        im_pred = im.copy()
        ground_truth = annotations[filename.replace('.jpg', '.png')]
        for gt in ground_truth:
            im = cv2.rectangle(im, (gt[1], gt[0]), (gt[3], gt[2]), (0, 255, 200), 1)
        # print("gt: ", ground_truth)
        predictions = predictions.detach().to('cpu').numpy()
        for i in range(predictions.shape[0]):
            im_pred = cv2.rectangle(im_pred, tuple(predictions[i, [0, 1]].astype(int)), tuple(predictions[i, [2, 3]].astype(int)), (0, 255, 200), 1)
        cv2.imshow('Ground Truth', im)
        cv2.imshow('Prediction', im_pred)
        cv2.waitKey(0)
        cv2.destroyAllWindows
