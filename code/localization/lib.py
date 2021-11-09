import os
import glob
import json
import cv2
import matplotlib
matplotlib.use('tkagg')  # not use PyQt for matplotlib

import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image


files_to_ignore = ["patch_33796d37f9186dc8e9510a5c5936deca_X20.0.png"]
path_models = 'runs/train/'


def json_2_yolo(path_json, path_data, img_fmt='.png'):
    """
    Converts json bounding boxes labels into Yolo-readable format.
    Will create one txt file for each image and save it in the same
    location.

    Inputs:
        path_json (str): path to json file
        path_dat (str): path to images folder
    Returns:
        None
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


def convert_color(path_in, path_out, conversion, fmt='.png'):
    """
    Converts all images of a folder to another color or colorspace.
    If needed will create a folder a save converted images into it.

    Inputs:
        path_in (str): path of input images
        path_out (str): path for output images
        converison (str): required conversion (possible values: 'H', 'HSV', 'gray')
        fmt (str): input image format
    Returns:
        None
    """
    img_files = glob.glob(path_in + '/*' + fmt)
    # print(img_files)
    # creating output folder
    if not os.path.exists(path_out):
        print("Creating output folder")
        os.makedirs(path_out)
    for filepath in img_files:
        img = cv2.imread(filepath, 1)
        if conversion == 'HSV':
            # convert to HSV
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif conversion == 'H':
            # keep Hue value
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
        elif conversion == 'gray':
            # convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif conversion:  # unknwon parameter convert
            raise ValueError("param unknown: {}".format(conversion))
        cv2.imwrite(path_out + os.path.basename(filepath), img)


def plot_results(model_name=None, show=True, save=False, path_save=None):
    """
    Plot results of Yolo model. Will plot a subplots of losses and metrics along epochs.

    Inputs:
        model_name (str): Yolo model name
        show (bool): if True, will show output results
        save (bool): if True, will save output plot to path_save
        path_save (str): path to save plot
    Returns:
        None
    """
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    results_path = path_models + model_name + '/results.csv'
    # load results
    data = pd.read_csv(results_path)
    metric_names = [x.strip() for x in data.columns]
    print("Best mAP:", data.loc[:, '     metrics/mAP_0.5'].max())
    x = data.values[:, 0]
    # plot
    for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
        y = data.values[:, j]
        ax[i].plot(x, y, marker='.', linewidth=2, markersize=8)
        ax[i].set_title(metric_names[j], fontsize=12)
        ax[i].grid()
    plt.grid()
    if show:
        plt.show()
    if save:
        fullpath_save = path_save + '/plots/' + model_name + '_results.pdf'
        print("Saving fig at: {}".format(fullpath_save))
        fig.savefig(fullpath_save)
    return data
