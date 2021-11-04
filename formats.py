import os
import json
from PIL import Image


files_to_ignore = ["patch_33796d37f9186dc8e9510a5c5936deca_X20.0.png"]


def json_2_yolo(path_json, path_data, img_fmt='.png'):
    """
    Converts json bounding boxes labels into Yolo-readable format.
    Will create one txt file for each image and save it in the same
    location.

    Inputs:
        path_json (str): path to json file
        path_dat (str): path to images folder
    """
    # read json file
    f = open(path_json, "r")
    annotations = json.loads(f.read())
    f.close()

    for img_filename, boxes in annotations.items():
        print(img_filename)
        # ignore wrong label files
        if img_filename in files_to_ignore:
            continue
        if boxes:
            # get image size
            im = Image.open(path_data + img_filename)
            width, height = im.size
            # create output file
            output_file = open(path_data + os.path.basename(img_filename).split(img_fmt)[0] + '.txt', "w+")
            # convert bouding box coordinates, Yolo format is [class, x, y,
            # width, height] with all values normalized
            for box in boxes:
                line = "0 {} {} {} {}\n".format(
                    (box[1] + box[3]) / (2 * height),
                    (box[0] + box[2]) / (2 * width),
                    (box[3] - box[1]) / height,
                    (box[2] - box[0]) / width
                )
                output_file.write(line)
            output_file.close()


if __name__ == '__main__':
    # path_json = "../data/localization/boxes_train.json"
    # path_data = "../data/localization/train/"
    path_json = "../data/localization/boxes_test.json"
    path_data = "../data/localization/test/"
    json_2_yolo(path_json, path_data, img_fmt='.png')
