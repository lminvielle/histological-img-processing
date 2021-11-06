import cv2
import glob

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from skimage.feature import local_binary_pattern, hog
from tqdm.auto import tqdm


path_data = '../data/classification/'


class Data:
    """
    Data manipulation.
    """

    def __init__(self):
        self.train_files = glob.glob(path_data + 'train/nuclei/*.png') + glob.glob(path_data + 'train/no_nuclei/*.png')
        ### JUST TO TEST RAPIDLY ###
        # self.train_files = list(np.random.choice(self.train_files, size=200, replace=False))

        self.test_files = glob.glob(path_data + 'test/nuclei/*.png') + glob.glob(path_data + 'test/no_nuclei/*.png')

    def load(self):
        """
        """
        return (self.train_files, self.test_files)


class Features:
    """
    Features for image processing.
    """

    def __init__(self, feat_list, feat_params, preprocess_params):
        self.feat_list = feat_list
        self.feat_params = feat_params
        self.preprocess_params = preprocess_params

    def preprocess(self, filepath, **params):
        dim_resize = params['dim_resize']
        im = cv2.imread(filepath, 1)
        # resize
        im = cv2.resize(im, dsize=dim_resize, interpolation=cv2.INTER_AREA)
        if params['convert'] == 'to_gray':
            # convert to grayscale
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        elif params['convert'] == 'to_H':
            # keep Hue value
            im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 0]
        elif params['convert']:
            raise ValueError("param 'convert' unknown")
        return im

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

    def hog(self, img, **params):
        """
        Returns the Histogram of Gradient of an input image
        """
        pix_per_cell = params['pixels_per_cell']
        feat = hog(img, orientations=8, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(1, 1), visualize=False)
        return feat

    def color_hist(self, img, **params):
        """
        """
        bins = params['bins']
        if img.ndim == 3:
            hist = []
            for clr_dim in range(img.shape[2]):
                hist_i, _ = np.histogram(img[:, :, clr_dim].ravel(), bins=bins, density=True)
                hist += list(hist_i)
            return np.array(hist)
        else:
            return np.histogram(img.ravel(), bins=bins, density=True)[0]

    def compute_X(self, files):
        """
        Returns features over a list of image files.
        """
        X = []
        for filepath in tqdm(files, ncols=80):
            features = []
            im = self.preprocess(filepath, **self.preprocess_params)
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
        return np.array(X)

    def compute_y(self, files):
        """
        Returns labels of a list of nuclei image files
        """
        y = []
        for filepath in files:
            if 'no_nuclei' in filepath:
                y.append(0)
            else:
                y.append(1)
        return np.array(y)

    def compute_Xy(self, files):
        """
        """
        print("Compute features...")
        return(self.compute_X(files), self.compute_y(files))


class Metrics:
    """
    """

    def roc_auc(self, y_true, y_pred):
        """
        Computes the aera under the ROC curve
        """
        self.roc_auc_ = roc_auc_score(y_true, y_pred)
        return self.roc_auc_

    def accuracy(self, y_true, y_pred):
        self.accuracy_ = accuracy_score(y_true, y_pred)
        return self.accuracy_


class Plots:
    """
    """

    def plot_roc(self, y_true, y_pred, features, show=True, save=False, path_save=None, return_outputs=False):
        fpr, tpr, _ = roc_curve(y_true, y_pred, drop_intermediate=False)
        fig, ax = plt.subplots(num="ROC curve")
        self.roc_auc(y_true, y_pred)
        ax.plot(fpr, tpr, label="AUC = {:.2f}".format(self.roc_auc_))
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.legend(loc="lower right")
        if show:
            plt.show()
        if save:
            name_save = "ROC_{}_{}".format(features.preprocess_params['convert'], '-'.join(features.feat_list))
            print('saving figure to:', path_save + name_save)
            fig.savefig(path_save + name_save + '.pdf')
        if return_outputs:
            return (fpr, tpr)

    def plot_feat_imp(self):
        if not self.model_name == 'rf':
            print("Cannot compute feature importance with this type of classifier !")
            return 0
        feat_imp = self.model.feature_importances_
        fig, ax = plt.subplots(num="Feature importance")
        ax.bar()
        plt.show()

    def plot_classification_results(self, files, y_true, y_pred, class_names=None):
        """
        y_pred must be of ints (not probabilities)
        """
        for i_file, filepath in enumerate(files):
            y_pred[i_file]
            truth_str = "Truth: {}".format(class_names[y_true[i_file]])
            predict_str = "Predicted: {}".format(class_names[y_pred[i_file]])
            im = cv2.imread(filepath, 1)
            cv2.putText(im, truth_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 80, 0), 2)
            cv2.putText(im, predict_str, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 180, 0), 2)
            cv2.imshow("Prediction", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


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
        if not self.is_fitted:
            print("Model is not fitted yet !")
            return 0
        if self.n_classes == 2:
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict_proba(X)


if __name__ == '__main__':
    my_data = Data()
    train_files, test_files = my_data.load()

    image = cv2.imread(train_files[0], 1)
    feat, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1), visualize=True)
    # channels = [0, 1, 2]
    # histsize = [25, 25, 25]
    # ranges = [0, 256]
    # clr_hist = cv2.calcHist(image, channels, mask=None, histSize=histsize, ranges=ranges)
    # hist, _ = np.histogram(image[:, :, 0].ravel(), bins=20, range=None, normed=None, weights=None, density=True)
    # fig, ax = plt.subplots()
    # ax.hist(image[:, :, 0].ravel(), bins=20, density=True)
    # plt.show()
    feat_list = [
        # 'lbp',
        # 'hog',
        'color_hist',
    ]
    feat_params = {
        # 'lbp': {'radius': 3},
        # 'hog': {'pixels_per_cell': 16},
        'color_hist': {'bins': 20},
    }
    feat = Features(feat_list, feat_params)
    clr_hist = feat.color_hist(image, bins=20)
    fig, ax = plt.subplots()
    ax.bar(np.arange(clr_hist.shape[0]), height=clr_hist)
    plt.show()
    cv2.imshow('im', image)
    # cv2.imshow('hog', hog_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
