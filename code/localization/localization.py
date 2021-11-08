import cv2
import glob
import json
import os

from yolov5 import train, YOLOv5

import numpy as np

# local lib
from lib import plot_results

# =======================================================
#   Params
# =======================================================
TRAIN = 0
TEST = 1

path_data = '../../../data/localization/'
path_save = '../../outputs/localization/'
path_models = 'runs/train/'

# =======================================================
#   Processing
# =======================================================
if TRAIN:
    train.run(imgsz=256, batch_size=16, epochs=500, weights='yolov5l.pt', data='data.yaml', name='sizeL_clrHSV')


if TEST:
    model_name = 'sizeL_clrHSV'
    path_imgs = path_data + '/test_HSV/'

    # plot results
    # data = plot_results(model_name, show=1, save=1, path_save=path_save)

    files_test = glob.glob(path_imgs + '*.png')
    if len(files_test) == 0:
        raise ValueError("Error: files_test is empty")

    # predict
    yolo = YOLOv5(path_models + model_name + '/weights/best.pt')
    results = yolo.predict(files_test)
    # read json file
    path_json = path_data + 'boxes_test.json'
    f = open(path_json, "r")
    annotations = json.loads(f.read())
    f.close()

    # draw predictions
    color_gt = (20, 200, 20)
    color_pred = (200, 200, 20)
    for filename, im, predictions in zip(results.files, results.imgs, results.pred):
        print(filename)
        # convert back to RGB if necessary
        if 'HSV' in model_name:
            # print('convert')
            # invert first and third channels
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            # go back to BGR
            im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
        im_pred = im.copy()
        # draw ground truth boxes
        ground_truth = annotations[filename.replace('.jpg', '.png')]
        for gt in ground_truth:
            im = cv2.rectangle(im, (gt[1], gt[0]), (gt[3], gt[2]), color_gt, 2)
        # print("gt: ", ground_truth)
        # draw predicted boxes
        predictions = predictions.detach().to('cpu').numpy()
        for i in range(predictions.shape[0]):
            im_pred = cv2.rectangle(im_pred, tuple(predictions[i, [0, 1]].astype(int)), tuple(predictions[i, [2, 3]].astype(int)), color_pred, 2)
        # cv2.putText(im, "Ground Truth", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # cv2.putText(im_pred, "Pred", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite(path_save + '/images/' + filename, im)
        cv2.imwrite(path_save + '/images/' + os.path.basename(filename) + '_pred.png', im_pred)
        # cv2.imshow('Ground Truth', im)
        # cv2.imshow('Prediction', im_pred)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows
