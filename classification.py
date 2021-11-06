import glob
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

from lib import ClassificationModel, Data, Features, Metrics, Plots

# load data
my_data = Data()
train_files, test_files = my_data.load()
# set features params
feat_list = [
    # 'lbp',
    'hog',
]
feat_params = {
    'lbp': {'radius': 3},
    'hog': {'pixels_per_cell': 16}
}
# compute features
features = Features(feat_list, feat_params)
X_train, y_train = features.compute_Xy(train_files)
X_test, y_test = features.compute_Xy(test_files)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=True)
# fit model on train set
clf_model = ClassificationModel('rf')
clf_model.fit(X_train, y_train)
# compute results
y_pred = clf_model.predict_proba(X_train)
score = clf_model.roc_auc(y_train, y_pred)
print("ROC AUC score train: {}".format(score))
# y_pred = clf_model.predict_proba(X_val)
# score = clf_model.roc_auc(y_val, y_pred)
# print("ROC AUC score val: {}".format(score))
y_pred = clf_model.predict_proba(X_test)
score = clf_model.roc_auc(y_test, y_pred)
print("ROC AUC score test: {}".format(score))
# clf_model.plot_roc(y_test, y_pred)

# plot results
# clf_model.plot_classification_results(test_files, y_test, y_pred > 0.5, class_names=['NO_nuclei', 'Nuclei'])
