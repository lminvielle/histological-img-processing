import os
import json
from PIL import Image

# prepare data json to yolo

# path_json = "../data/localization/boxes_train.json"
# path_data = "../data/localization/train/"
path_json = "../data/localization/boxes_test.json"
path_data = "../data/localization/test/"
f = open(path_json, "r")
annotations = json.loads(f.read())

for img_filename, boxes in annotations.items():
    if boxes:
        im = Image.open(path_data + img_filename)
        print(im.size)
        width, height = im.size
        output_file = open(path_data + os.path.basename(img_filename).split('.')[0] + '.txt', "w+")
        for box in boxes:
            line = "0 {} {} {} {}\n".format(
                (box[1] + box[3]) / (2 * height),
                (box[0] + box[2]) / (2 * width),
                (box[3] - box[1]) / height,
                (box[2] - box[0]) / width
            )
            print('line: ', line)
            output_file.write(line)
        output_file.close()
