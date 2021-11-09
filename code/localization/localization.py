import cv2
import glob
import json
import os

from yolov5 import train, YOLOv5

# local lib
from lib import json_2_yolo, convert_color, plot_results

# =======================================================
#   Params
# =======================================================
path_data = '../../../data/localization/'

PREP_DATA = 0  # set to 1 to prepare data for Yolo training

TRAIN = 0  # set to 1 to launch testing
# train params
yolo_yaml_file = 'data.yaml'
batch_size = 16
epochs = 500
model_name = 'sizeL_clrRGB'  # replace RGB with HSV if trained over HSV images
weights = 'yolov5l.pt'  # architecure size: 'yolov5s.pt, 'yolov5m.pt', 'yolov5l.pt'

TEST = 1  # set to 1 to launch training
# test params
path_save = '../../outputs/localization/'
path_models = 'runs/train/'
path_imgs = path_data + '/test/'

model_name = 'sizeL_clrRGB'

# =======================================================
#   Processing
# =======================================================
if PREP_DATA:
    # if needed convert images to HSV colorspace and save them
    # path_data_convert = path_data + "/train_HSV/"
    # convert_color(path_data + '/train/', path_data_convert, 'HSV')
    # convert json labels to yolo labels
    json_2_yolo(path_data + '/boxes_train.json', path_data + '/train/', img_fmt='.png')
    json_2_yolo(path_data + '/boxes_test.json', path_data + '/test/', img_fmt='.png')

if TRAIN:
    train.run(imgsz=256, batch_size=batch_size, epochs=epochs, weights=weights, data=yolo_yaml_file, name=model_name)

if TEST:
    # plot results
    data = plot_results(model_name, show=1, save=1, path_save=path_save)

    # test image set
    files_test = glob.glob(path_imgs + '*.png')
    if len(files_test) == 0:
        raise ValueError("Error: files_test is empty")

    # load model & predict
    yolo = YOLOv5(path_models + model_name + '/weights/best.pt')
    results = yolo.predict(files_test)

    # read json file for ground truth annotations
    path_json = path_data + 'boxes_test.json'
    f = open(path_json, "r")
    annotations = json.loads(f.read())
    f.close()

    # draw predictions
    color_gt = (20, 200, 20)
    color_pred = (200, 200, 20)
    for filename, im, predictions in zip(results.files, results.imgs, results.pred):
        # yolo puts 'jpg' to results filenames: get it back to png
        filename = filename.replace('.jpg', '.png')
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
        ground_truth = annotations[filename]
        for gt in ground_truth:
            im = cv2.rectangle(im, (gt[1], gt[0]), (gt[3], gt[2]), color_gt, 1)
        # draw predicted boxes
        predictions = predictions.detach().to('cpu').numpy()
        for i in range(predictions.shape[0]):
            im_pred = cv2.rectangle(im_pred, tuple(predictions[i, [0, 1]].astype(int)), tuple(predictions[i, [2, 3]].astype(int)), color_pred, 1)
        cv2.imwrite(path_save + '/images/' + filename, im)
        cv2.imwrite(path_save + '/images/' + os.path.basename(filename).replace('.png', '_pred.png'), im_pred)
        # cv2.imshow('Ground Truth', im)
        # cv2.imshow('Prediction', im_pred)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows
