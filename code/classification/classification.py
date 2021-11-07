# import matplotlib
# matplotlib.use('tkagg')  # do not use PyQt for matplotlib

from lib import ClassificationModel, Data, Features

# =======================================================
#   Params
# =======================================================
# features params
feat_list = [
    'lbp',
    # 'hog',
    # 'color_hist',
]
feat_params = {
    'lbp': {'radius': 3, 'color': 'H'},
    'hog': {'pixels_per_cell': 16, 'color': 'H'},
    'color_hist': {'bins': 20, 'color': 'BGR'},
}
# preprocessing
preprocess_params = {
    'dim_resize': (256, 256),
}
path_save = '../outputs/classification/'

# =======================================================
#   Processing
# =======================================================
# load data
my_data = Data()
train_files, test_files = my_data.load()
# compute features
features = Features(feat_list, feat_params, preprocess_params)
X_train, y_train = features.compute_Xy(train_files)
X_test, y_test = features.compute_Xy(test_files)
# fit model on train set
clf_model = ClassificationModel('rf')
clf_model.fit(X_train, y_train)
# compute results
# on train set
y_pred = clf_model.predict_proba(X_train)
score = clf_model.roc_auc(y_train, y_pred)
print("ROC AUC score train: {}".format(score))
# on test set
y_pred = clf_model.predict_proba(X_test)
score = clf_model.roc_auc(y_test, y_pred)
print("ROC AUC score test: {}".format(score))
score = clf_model.accuracy(y_test, y_pred > 0.5)
print("Accuracy test: {}".format(score))
# roc curve
clf_model.plot_roc(y_test, y_pred, features, show=True, save=False, path_save=path_save)

# plot results
# clf_model.plot_classification_results(test_files, y_test, y_pred > 0.5, class_names=['NO_nuclei', 'Nuclei'])
