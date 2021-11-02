import glob
import os
# from PIL import Image
import matplotlib
matplotlib.use('tkagg')  # do not use PyQt for matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm.auto import tqdm
from collections import Counter
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from skimage.feature import local_binary_pattern

path_data = '../data/classification/'
imgs_train = glob.glob(path_data + 'train/nuclei/*.png') + glob.glob(path_data + 'train/no_nuclei/*.png')
imgs_test = glob.glob(path_data + 'test/nuclei/*.png') + glob.glob(path_data + 'test/no_nuclei/*.png')
print('size train set: ', len(imgs_train))

REDUCE_TRAIN = 0

if REDUCE_TRAIN:
    imgs_train = np.random.choice(imgs_train, size=200, replace=False)


X_train = []
y_train = []
classes = ['NO Nuclei', 'Nuclei']

radius = 3
n_points = 8 * radius
n_bins = n_points + 2
range_hist = n_points + 2

for filepath in tqdm(imgs_train, ncols=80):
    # filepath = imgs_train[1]
    im = cv2.imread(filepath, 1)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # keep Hue value
    im_h = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 0]
    # cv2.imshow('Image', im)
    # cv2.imshow('Image gray', im_gray)
    # cv2.imshow('Image H', im_h)

    lbp = local_binary_pattern(im_h, n_points, radius, 'uniform')
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, range_hist))
    X_train.append(hist)
    # cv2.imshow('Features', lbp)
    # fig, ax = plt.subplots()
    # ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, range_hist), facecolor='0.5')
    # hist(ax, hist)

    # cv2.imshow('Features', imgg)
    # if des is None:
        # print('could not find ORB')
        # continue
    # X_train.append(des[:100, :])
    if 'no_nuclei' in filepath:
        y_train.append(0)
    else:
        y_train.append(1)

    # cv2.waitKey(0)
    # plt.show()
    # cv2.destroyAllWindows()


print('Training data count: {}'.format(Counter(y_train)))

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train_train, X_test, y_train_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

clf = ensemble.RandomForestClassifier(
    n_estimators=50,
    oob_score=False,
    max_depth=None,
    class_weight=None)

clf.fit(X_train_train, y_train_train)
predict_proba_train = clf.predict_proba(X_train)
predict_proba_test = clf.predict_proba(X_test)
score_train = roc_auc_score(y_train, predict_proba_train[:, 1])
score_test = roc_auc_score(y_test, predict_proba_test[:, 1])
print("score train: {}".format(score_train))
print("score test: {}".format(score_test))


# fit on whole train set
clf.fit(X_train, y_train)

# test on test folder
X_test = []
y_test = []
for filepath in tqdm(imgs_test, ncols=80):
    im_clr = cv2.imread(filepath, 1)
    im_gray = cv2.cvtColor(im_clr, cv2.COLOR_BGR2GRAY)
    # keep Hue value
    im_h = cv2.cvtColor(im_clr, cv2.COLOR_BGR2HSV)[:, :, 0]

    lbp = local_binary_pattern(im_h, n_points, radius, 'uniform')
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, range_hist))
    X_test.append(hist)

    if 'no_nuclei' in filepath:
        y_test.append(0)
    else:
        y_test.append(1)

    # Plot result
    # predict = clf.predict(hist .reshape((1, -1)))
    # truth_str = "Truth: {}".format(classes[y_test[-1]])
    # predict_str = "Predicted: {}".format(classes[predict[0]])
    # cv2.putText(im_clr, truth_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 80, 0), 2)
    # cv2.putText(im_clr, predict_str, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 180, 0), 2)
    # cv2.imshow("prediction", im_clr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

X_test = np.array(X_test)
y_test = np.array(y_test)
print('Test data count: {}'.format(Counter(y_test)))
predict_proba = clf.predict_proba(X_test)
score = roc_auc_score(y_test, predict_proba[:, 1])
print("ROC AUC score test folder: {}".format(score))
score = accuracy_score(y_test, np.argmax(predict_proba, axis=1))
print("Accuracy score test folder: {}".format(score))
