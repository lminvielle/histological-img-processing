import os
import cv2
import glob

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from skimage.feature import local_binary_pattern, hog
from tqdm.auto import tqdm


path_data = '../../../data/classification/'


class Data:
    """
    Data manipulation.
    """

    def __init__(self):
        self.train_files = glob.glob(path_data + 'train/nuclei/*.png') + glob.glob(path_data + 'train/no_nuclei/*.png')

        self.test_files = glob.glob(path_data + 'test/nuclei/*.png') + glob.glob(path_data + 'test/no_nuclei/*.png')
        if len(self.train_files) == 0:
            raise FileNotFoundError("No image found in path")

    def load(self):
        """
        Returns training and testing files

        Inputs:
            None
        Returns:
            train_files (list): training files
            test_files (list): testing files
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

    def get_name_save(self):
        """
        Returns a string corresponding to features and there parameters

        Inputs:
            None
        Returns:
            name (str)
        """
        name = ''
        for i_feat, feat in enumerate(self.feat_list):
            name += feat + '_'
            for key, el in self.feat_params[feat].items():
                name += key + '-'
                name += str(el)
            if i_feat < (len(self.feat_list) - 1):
                name += '_'
        return name

    def preprocess(self, filepath, **params):
        """
        Preprocress a input image.

        Inputs:
            filepath: path to input image
            **params: keyword arguments for preprocessing:
                dim_rsize (tuple): desired output size
        """
        dim_resize = params['dim_resize']
        img = cv2.imread(filepath, 1)
        # resize
        img = cv2.resize(img, dsize=dim_resize, interpolation=cv2.INTER_AREA)
        return img

    def convert(self, img, conversion):
        """
        Converts the colorspace of an image.

        Inputs:
            img (Opencv image): image
            conversion (str): desired conversion. Possible values are: 'HSV', 'H', 'gray'
        """
        if conversion == 'HSV':
            # convert to HSV
            # print('hsv')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif conversion == 'H':
            # keep Hue value
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
        elif conversion == 'gray':
            # convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif conversion:  # unknwon parameter convert
            raise ValueError("param unknown: {}".format(conversion))
        return img

    def lbp(self, img, **params):
        """
        Returns the histogram of Local Binary Patterns of an image.

        Inputs:
            img (Opencv image): image
            **params: keyword arguments:
                radius (int): radius of the LBP
        Returns:
            hist (Numpy array)
        """
        # convert only if not BGR wanted
        if params['color'] and params['color'] != 'BGR':
            img = self.convert(img, params['color'])
        # params
        radius = params['radius']
        n_points = 8 * radius
        n_bins = n_points + 2
        range_hist = n_points + 2
        # lbp
        lbp = local_binary_pattern(img, n_points, radius, 'uniform')
        # compute hist over lbp
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, range_hist))
        return hist

    def hog(self, img, **params):
        """
        Returns the Histogram of Gradient of an input image

        Inputs:
            img (Opencv image): image
            **params: keyword arguments:
                color (str): color conversion before applying HOG
                pixels_per_cell (str): number of pixels per cell
        Returns:
            feat (Numpy array)
        """
        # convert only if not BGR wanted
        if params['color'] and params['color'] != 'BGR':
            img = self.convert(img, params['color'])
        # hog params
        pix_per_cell = params['pixels_per_cell']
        # hog
        feat = hog(img, orientations=8, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(1, 1), visualize=False)
        return feat

    def color_hist(self, img, **params):
        """
        Returns the color histogram of input image.

        Inputs:
            img (Opencv image): image
            **params: keyword arguments:
                color (str): color conversion before applying HOG
                bins (int): number of bins of the output hist
        Returns:
            hist (Numpy array)
        """
        # convert only if not BGR wanted
        if params['color'] and params['color'] != 'BGR':
            img = self.convert(img, params['color'])
        # hist params
        bins = params['bins']
        if img.ndim == 3:
            hist = []
            for clr_dim in range(img.shape[2]):
                hist_i, _ = np.histogram(img[:, :, clr_dim].ravel(), bins=bins, density=True)
                hist += list(hist_i)
            return np.array(hist)
        else:
            # print("only one channel")
            return np.histogram(img.ravel(), bins=bins, density=True)[0]

    def compute_X(self, files):
        """
        Returns features over a list of image files.
        Inputs:
            files (list): input files to be processed
        Returns:
            X (Numpy array)
        """
        X = []
        for filepath in tqdm(files, ncols=80):
            features = []
            # image preprocessing
            im = self.preprocess(filepath, **self.preprocess_params)
            for feat_name in self.feat_list:
                # get feature function name
                feat_function = getattr(self, feat_name)
                # compute feature
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
        Inputs:
            files (list): input files to be processed
        Returns:
            y (Numpy array)
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
        Returns feature matrix and labels given a list of files

        Inputs:
            files (list): input files to be processed
        Returns:
            (X, y) (Numpy arrays)
        """
        print("Compute features...")
        return(self.compute_X(files), self.compute_y(files))


class Metrics:
    """
    Classification metrics
    """

    def roc_auc(self, y_true, y_pred):
        """
        Computes the aera under the ROC curve
        Inputs:
            y_true (Numpy array of int): array of true class
            y_pred (Numpy array if float): array of predicted probabilites. MUST BE of floats.
        Return:
            auc (float): ROC AUC
        """
        self.roc_auc_ = roc_auc_score(y_true, y_pred)
        return self.roc_auc_

    def accuracy(self, y_true, y_pred):
        """
        Returns the accuracy given true and predicted class

        Inputs:
            y_true (Numpy array of int): array of true class
            y_pred (Numpy array if float): array of predicted classes. MUST BE of ints.
        Return:
            accuracy (float)
        """
        self.accuracy_ = accuracy_score(y_true, y_pred)
        return self.accuracy_


class Plots:
    """
    For plotting results
    """

    def plot_roc(self, y_true, y_pred, features, show=True, save=False, path_save=None, return_outputs=False):
        """
        Plots ROC curve given results of a classifer over a test set.

        Inputs:
            y_true (Numpy array of int): array of true class
            y_pred (Numpy array if float): array of probabilites. MUST BE of floats.
            features (instance of Features): features with which the model was trained
            show (bool): if True, will show output plot
            save (bool): if True, will save output plot to path_save
            path_save (str): path to save plot
            return_outputs (bool): if True, return computed fpr and tpr
        Returns:
            None or (fpr, tpr)
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred, drop_intermediate=False)
        plt.rcParams.update({'font.size': 22})
        fig, ax = plt.subplots(num="ROC curve")
        self.roc_auc(y_true, y_pred)
        ax.plot(fpr, tpr, label="AUC = {:.2f}".format(self.roc_auc_))
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.legend(loc="lower right")
        plt.subplots_adjust(left=0.19, bottom=0.17)
        if show:
            plt.show()
        if save:
            name_save = "ROC_{}".format(features.get_name_save())
            print('saving figure to:', path_save + name_save)
            fig.savefig(path_save + name_save + '.pdf')
        if return_outputs:
            return (fpr, tpr)

    # def plot_feat_imp(self):
        # if not self.model_name == 'rf':
            # print("Cannot compute feature importance with this type of classifier !")
            # return 0
        # feat_imp = self.model.feature_importances_
        # fig, ax = plt.subplots(num="Feature importance")
        # ax.bar()
        # plt.show()

    def plot_classification_results(self, files, y_true, y_pred, class_names=None, show=True, save=False, path_save=None):
        """
        Plots classification results as images with ground truth and prediction
        written in it.

        Inputs:
            files (list):
            y_true (Numpy array of int): array of true class
            y_pred (Numpy array if int): array of predicted class. MUST BE of ints.
            class_names (list): class names corresponding to y_true & y_pred
            show (bool): if True, will show output images
            save (bool): if True, will save output images to path_save
            path_save (str): path to save images
        Returns:
            None
        """
        if save:
            print("Will try and save images with prediction drawn, to {}".format(path_save))
        for i_file, filepath in enumerate(files):
            # get true and predicted classes
            truth_str = "Truth: {}".format(class_names[y_true[i_file]])
            predict_str = "Predicted: {}".format(class_names[y_pred[i_file]])
            # good or bad prediction
            if y_pred[i_file] != y_true[i_file]:
                is_wrong = True
                color = (20, 20, 200)
            else:
                is_wrong = False
                color = (20, 180, 0)
            # read input image and write prediction on it
            im = cv2.imread(filepath, 1)
            cv2.putText(im, truth_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 80, 0), 2)
            cv2.putText(im, predict_str, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if show:
                cv2.imshow("Prediction", im)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if save:
                name_save = os.path.basename(filepath).split('.png')[0] + ('_WRONG' if is_wrong else '_ok') + '.png'
                cv2.imwrite(path_save + name_save, im)


class ClassificationModel(Metrics, Plots):
    """
    Classification model
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

        Inputs:
            X (Numpy array): feature matrix
            y (Numpy array): labels
        Returns:
            None
        """
        self.model.fit(X, y)
        self.is_fitted = True
        self.n_classes = len(set(y))

    def predict_proba(self, X):
        """
        Returns the model prediction over input data.

        Inputs:
            X (Numpy array): feature matrix
        Returns:
            predict_proba: (Numpy array): array of estimated probabilities
        """
        if not self.is_fitted:
            print("Model is not fitted yet !")
            return 0
        if self.n_classes == 2:
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict_proba(X)
