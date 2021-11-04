import cv2
import glob

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from skimage.feature import local_binary_pattern
from tqdm.auto import tqdm


path_data = '../data/classification/'


class Data:
    """
    Data manipulation.
    """

    def __init__(self):
        self.train_files = glob.glob(path_data + 'train/nuclei/*.png') + glob.glob(path_data + 'train/no_nuclei/*.png')
        ### JUST TO TEST RAPIDLY ###
        self.train_files = np.random.choice(self.train_files, size=200, replace=False)

        self.test_files = glob.glob(path_data + 'test/nuclei/*.png') + glob.glob(path_data + 'test/no_nuclei/*.png')

    def load(self):
        """
        """
        return (self.train_files, self.test_files)


class Features:
    """
    Features for image processing.
    """

    def __init__(self, feat_list, feat_params):
        self.feat_list = feat_list
        self.feat_params = feat_params

    def lbp(self, img, **params):
        """
        Returns the Local Binary Patterns of an image.
        """
        radius = params['radius']
        n_points = 8 * radius
        n_bins = n_points + 2
        range_hist = n_points + 2
        lbp = local_binary_pattern(img, n_points, radius, 'uniform')
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, range_hist))
        return hist

    def compute(self, files):
        """
        Returns computes features over a list of image files.
        """
        X = []
        y = []
        for filepath in tqdm(files, ncols=80):
            im = cv2.imread(filepath, 1)
            # keep Hue value
            im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 0]
            features = []
            for feat_name in self.feat_list:
                feat_function = getattr(self, feat_name)
                feature = feat_function(im, **self.feat_params[feat_name])
                if type(feature) == list:
                    features += feature
                elif type(feature) == np.ndarray:
                    features += list(feature)
                else:
                    features.append(feature)

            X.append(features)
            if 'no_nuclei' in filepath:
                y.append(0)
            else:
                y.append(1)

        return (np.array(X), np.array(y))


class Metrics:
    """
    """

    def roc_auc(self, y_true, y_pred):
        """
        Computes the aera under the ROC curve
        """
        if not self.is_fitted:
            print("Model is not fitted yet !")
            return 0
        self.roc_auc_ = roc_auc_score(y_true, y_pred)
        return self.roc_auc_

    def accuracy(self, y_true, y_pred):
        self.accuracy_ = accuracy_score(y_true, y_pred)
        return self.accuracy_


class Plots:
    """
    """
    def plot_roc(self, y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred, drop_intermediate=False)
        fig, ax = plt.subplots(num="ROC curve")
        self.roc_auc(y_true, y_pred)
        ax.plot(fpr, tpr, label="AUC = {:.2f}".format(self.roc_auc_))
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.legend(loc="lower right")
        plt.show()

    def plot_feat_imp(self):
        if not self.model_name == 'rf':
            print("Cannot compute feature importance with this type of classifier !")
            return 0
        feat_imp = self.model.feature_importances_
        fig, ax = plt.subplots(num="Feature importance")
        plt.show()


class ClassificationModel(Metrics, Plots):
    """
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.is_fitted = False
        self.init_model()

    def init_model(self):
        """
        Creates a classification model.
        """
        if self.model_name == 'rf':
            self.model = ensemble.RandomForestClassifier(
                n_estimators=50,
                oob_score=False,
                max_depth=None,
                class_weight=None)

    def fit(self, X, y):
        """
        Fits the classification model.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        self.n_classes = len(set(y))

    def predict_proba(self, X):
        """
        Returns the model prediction over input data.
        """
        if self.n_classes == 2:
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict_proba(X)


if __name__ == '__main__':
    my_data = Data()
    train_files, test_files = my_data.load()

    feat_list = ['lbp']
    feat_params = {
        'lbp': {'radius': 3}
    }
    features = Features(feat_list, feat_params)
    X_train, y_train = features.compute(train_files)
    X_test, y_test = features.compute(test_files)
    clf_model = ClassificationModel('rf')
    clf_model.fit(X_train, y_train)

    y_pred = clf_model.predict_proba(X_train)
    score = clf_model.roc_auc(y_train, y_pred)
    print("ROC AUC score train: {}".format(score))
    y_pred = clf_model.predict_proba(X_test)
    score = clf_model.roc_auc(y_test, y_pred)
    print("ROC AUC score test: {}".format(score))
    clf_model.plot_roc(y_test, y_pred)
