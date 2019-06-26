import csv
import os
import random
import traceback
import warnings
from itertools import cycle

import h5py
import jsonpickle
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy import interp
from sklearn import datasets, svm
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             classification_report, confusion_matrix,
                             make_scorer, precision_recall_curve, roc_curve)
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_predict, cross_val_score,
    train_test_split)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.multiclass import type_of_target

from cbh import config
from cbh.dumb import get_xkcd_colors

colorz = get_xkcd_colors()
colors = cycle(colorz['hex'])

# print(colorz['hex'])
# print(colorz['names'])

# resources
# https://github.com/javaidnabi31/Multi-class-with-imbalanced-dataset-classification/blob/master/20-news-group-classification.ipynb
# https://mashimo.wordpress.com/2017/11/04/cross-validation/
# https://stackoverflow.com/questions/55591063/how-to-perform-smote-with-cross-validation-in-sklearn-in-python
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
# https://stackoverflow.com/questions/45332410/sklearn-roc-for-multiclass-classification
# https://stackoverflow.com/questions/47876999/how-to-compute-average-roc-for-cross-validated-for-multiclass
# https://jmetzen.github.io/2015-04-14/calibration.html # has multiclass examples
# https://www.kaggle.com/lct14558/imbalanced-data-why-you-should-not-use-roc-curve
# https://stats.stackexchange.com/questions/147175/how-is-the-confusion-matrix-reported-from-k-fold-cross-validation

# When training starts, certain metrics are often zero for a while, which throws a warning and clutters the terminal output
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
print("Loading", os.path.basename(__file__))

# Load data
data = pd.read_hdf(config.RAW_DATA_FILE_H5, key="data")
# print(list(data))

# Define labels and features, and binarize labels for AUC/PR curves
y = data["diagnosis"].copy()
X = data.drop(["diagnosis", "id"], axis=1).copy()
# print(X.head())
# print(y.head())
classes = np.unique(y)
# print(classes)
y_true = label_binarize(y, classes=classes)
n_classes = y_true.shape[1]

# Define cross-validation scheme
n_splits = 10
k = StratifiedKFold(n_splits=n_splits, random_state=config.SEED, shuffle=True)
kf = StratifiedKFold(n_splits=n_splits, random_state=config.SEED, shuffle=True).split(
    X, y
)

# Define classifier
params = config.PARAMS_LGBM
clf = lgb.LGBMClassifier(**params)

# Prepare for plotting
base_fpr = np.linspace(0, 1, 101)
plt.style.use("ggplot")
plt.figure(figsize=(12, 8))

fpr = dict()
tpr = dict()
roc_auc = dict()





# Start with empty lists for average classification report
originalclass = []
predictedclass = []

def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    report = classification_report(originalclass, predictedclass, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(config.METRIC_FIGS_DIR / "classification_report.csv")
    report_df.to_csv(config.TABLES_DIR / "classification_report.csv")
    return accuracy_score(y_true, y_pred)

scores = cross_val_score(clf, X=X, y=y, cv=k, \
               scoring=make_scorer(classification_report_with_accuracy_score))

# append to classification report
with open(config.METRIC_FIGS_DIR / "classification_report.csv",'a') as file:
    for i in range(n_splits):
        writer = csv.writer(file, delimiter=',')
        writer.writerow([f"Score for fold {i}",  f"{scores[i]}"])
    writer.writerow([f"Mean accuracy with standard deviation",  f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"])


# Initialize confusion matrix with zeros ...
conf_mats = np.zeros(shape=(len(classes), len(classes)))
# and empty dataframe to keep track of misclassified samples ...
misclassified_df = pd.DataFrame()
# and another for the roc curve data.
roc_curve_data = pd.DataFrame()

# Fit the model for each fold
# the "i" and "enumerate" give you the index for each loop of train and test
for i, (train, test) in enumerate(kf):
    print(f"\nFold {i}")
    model = clf.fit(X.iloc[train], y.iloc[train])
    y_score = model.predict_proba(X.iloc[test])
    # print(y_score)
    y_pred = model.predict(X.iloc[test])

    # Generate dataframes
    # Using a cutoff
    actual = y.iloc[test]
    successes = actual == y_pred
    predictions = pd.DataFrame(
        {
            "Success": successes,
            "Prediction": y_pred,
            "Actual": actual,
            "Sample Number": actual.index,
            "Iteration": i,
        }
    )
    predictions = predictions.reset_index(drop=True)

    # Using raw probabilities
    probs = pd.DataFrame(y_score, columns=[x.upper() for x in classes])
    probs = probs.reset_index(drop=True)

    # Combine dataframes
    combo = pd.concat([predictions, probs], axis=1)
    misclassified_df = misclassified_df.append(combo)
    misclassified_df = misclassified_df.reset_index(drop=True)
    # print(misclassified_df.tail(10))

    # Generate confusion matrix for this fold
    conf_mat = confusion_matrix(y.iloc[test], y_pred)

    # Add all the confusion matrices together for an overall picture
    conf_mats = conf_mats + conf_mat
    conf_mats = conf_mats.astype(int)
    b = sum(conf_mats)
    print("Number of predictions remaining:", len(y) - sum(b))


    # Plot final confusion matrix on last iteration
    if i == (n_splits-1):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            conf_mats,
            annot=True,
            fmt="d",
            xticklabels=[x.upper() for x in classes],
            yticklabels=[x.upper() for x in classes],
        )  # upper case the class names
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        # plt.show()
        plt.savefig(
            (config.METRIC_FIGS_DIR / "confusion_matrix.pdf"),
            dpi=1200,
            transparent=False,
            bbox_inches="tight",
        )
        plt.close()


    # Compute ROC curve and ROC area for each class PER FOLD
    for j in range(n_classes):
        # print(classes[j])
        fpr[j], tpr[j], _ = roc_curve(y_true[test][:, j], y_score[:, j])
        # print(fpr[j])
        # print(type(fpr[j]))
        roc_auc[j] = auc(fpr[j], tpr[j])
        fpr_save = {f"{i}_{classes[j]}_fpr": fpr[j]}
        tpr_save = {f"{i}_{classes[j]}_tpr": tpr[j]}
        auc_save = {f"{i}_{classes[j]}_auc": [roc_auc[j]]}
        # print(auc_save)
        fpr_df = pd.DataFrame(fpr_save)
        tpr_df = pd.DataFrame(tpr_save)
        auc_df = pd.DataFrame(auc_save)
        new = pd.concat([fpr_df, tpr_df, auc_df], axis=1) 
        roc_curve_data = pd.concat([roc_curve_data, new], axis=1)
        # roc_curve_data = roc_curve_data.reset_index(drop=True)
        # print(roc_curve_data.head(100))
        # print(fpr_df.head(100))
        # print(auc_df)
        # print(fpr_save)
        # print("AUC:", roc_auc[j])



# Save combined dataframes
misclassified_df.to_csv(config.METRIC_FIGS_DIR / "misclassified_cv.csv")
misclassified_df.to_csv(config.TABLES_DIR / "misclassified_cv.csv")
roc_curve_data.to_csv(config.METRIC_FIGS_DIR / "roc_curve_data.csv")

# roc_data = pd.read_csv(config.METRIC_FIGS_DIR / "roc_curve_data.csv")
# print(roc_data.head(100))

# for classname in classes:
#     # select the columns with the appropriate classname
#     df2 = roc_data.filter(regex=classname)
#     fprs = df2.filter(regex='fpr')
#     tprs = df2.filter(regex='tpr')
#     aucs = df2.filter(regex='auc')
#     print(fprs.head(100))
#     # print(df2.head(100))


# fpr[j], tpr[j], _ = roc_curve(y_true[test][:, j], y_score[:, j])
# roc_auc[j] = auc(fpr[j], tpr[j])
# fpr_save = {f"{i}_{classes[j]}_fpr": fpr[j]}
# tpr_save = {f"{i}_{classes[j]}_tpr": tpr[j]}
# auc_save = {f"{i}_{classes[j]}_auc": [roc_auc[j]]}

# # y_train_bin = label_binarize(y_train, classes=["mds", "cmml", "pmf", "et", "pv"])
# # y_test_bin = label_binarize(y_test, classes=["mds", "cmml", "pmf", "et", "pv"])


# i = 0
# y = label_binarize(y, classes=["mds", "cmml", "pmf", "et", "pv"])
# print(type_of_target(y))
# n_classes = y.shape[1]

# for train, test in k.split(X, y):
#     # y.iloc[test] = label_binarize(y.iloc[test], classes =["mds", "cmml", "pmf", "et", "pv"])
#     probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])

#     # Compute ROC curve and ROC area for each class
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for idx in range(n_classes):
#         fpr[idx], tpr[idx], _ = roc_curve(y_test[:, idx], y_score[:, idx])
#         roc_auc[idx] = auc(fpr[idx], tpr[idx])

#     fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, idx])
#     tprs.append(interp(mean_fpr[idx], fpr[idx], tpr[idx]))
#     tprs[-1][0] = 0.0
#     roc_auc = auc(fpr[idx], tpr[idx])
#     aucs.append(roc_auc)
#     plt.plot(
#         fpr[idx],
#         tpr[idx],
#         lw=1,
#         alpha=0.3,
#         label=f"ROC fold %d (AUC = %0.2f)" % (i, roc_auc),
#     )

#     i += 1
# plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# plt.plot(
#     mean_fpr,
#     mean_tpr,
#     color="b",
#     label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
#     lw=2,
#     alpha=0.8,
# )

# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# plt.fill_between(
#     mean_fpr,
#     tprs_lower,
#     tprs_upper,
#     color="grey",
#     alpha=0.2,
#     label=r"$\pm$ 1 std. dev.",
# )

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver operating characteristic example")
# plt.legend(loc="lower right")
# plt.show()


# clf.fit(
#     X_train,
#     y_train,
#     eval_set=[(X_test, y_test)],
#     # eval_metric="logloss",
#     early_stopping_rounds=early_stopping_rounds,
# )


# def lgbm_save_model_to_h5():
#     """
#     To retrieve pickled model, use something like:
#     `pkl_model = metricsgen.save_model_to_pickle()`

#     then: 

#     `with open(pkl_model, "rb") as fin:
#             gbm_model = pickle.load(fin)` 
#     """

#     print("Dumping model with pickle...")
#     print("JSONpickling the model...")
#     frozen = jsonpickle.encode(clf)
#     print("Saving clf to .h5 file...")
#     h5_file = "test.h5"
#     with h5py.File(h5_file, "a") as f:
#         try:
#             f.create_dataset("clf", data=frozen)
#         except Exception as exc:
#             print(traceback.format_exc())
#             print(exc)
#             try:
#                 del f["clf"]
#                 f.create_dataset("clf", data=frozen)
#                 print("Successfully deleted old clf and saved new one!")
#             except:
#                 print("Old clf persists...")
#     print(h5_file)


# lgbm_save_model_to_h5()


# def lgbm_save_feature_importance_plot():
#     print("Plotting feature importances...")

#     ax = lgb.plot_importance(clf, figsize=(5, 20), importance_type="gain", precision=2)
#     plt.savefig(
#         (config.METRIC_FIGS_DIR / "feature_importance.pdf"),
#         dpi=1200,
#         transparent=False,
#         bbox_inches="tight",
#     )
#     plt.close()


# lgbm_save_feature_importance_plot()


# print(classification_report(y_test, clf.predict(X_test)))


# y = label_binarize(y, classes=[0, 1, 2])
# n_classes = 3

# # shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=0
# )

# # clf
# clf = OneVsRestClassifier(LinearSVC(random_state=0))
# y_score = clf.fit(X_train, y_train, method="predict_proba")

# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Plot of a ROC curve for a specific class
# for i in range(n_classes):
#     plt.figure()
#     plt.plot(fpr[i], tpr[i], label="ROC curve (area = %0.2f)" % roc_auc[i])
#     plt.plot([0, 1], [0, 1], "k--")
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("Receiver operating characteristic example")
#     plt.legend(loc="lower right")
#     plt.show()


# y_score = cross_val_predict(clf, X, y, cv=10, method="predict_proba")

# # reversefactor = dict(zip(range(5),definitions))
# # y_test = np.vectorize(reversefactor.get)(y_test)
# # y_pred = np.vectorize(reversefactor.get)(y_pred)

# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# lw = 2
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# colors = cycle(["blue", "red", "green", "aqua", "darkorange", "cornflowerblue"])
# # random_colors = random.sample(beau_colors, 20)
# colors = cycle(colorz['hex'])

# for i, color in zip(range(n_classes), colors):
#     target = "Missing"
#     if i == 0:
#         target = "MDS"
#     if i == 1:
#         target = "CMML"
#     if i == 2:
#         target = "PMF"
#     if i == 3:
#         target = "ET"
#     if i == 4:
#         target = "PV"
#     plt.plot(
#         fpr[i],
#         tpr[i],
#         color=color,
#         lw=lw,
#         label=f"ROC curve for {target} (area = {roc_auc[i]:0.2f})",
#     )
# plt.plot([0, 1], [0, 1], "k--", lw=lw)
# plt.xlim([-0.05, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("")
# plt.legend(loc="lower right")
# plt.savefig(config.METRIC_FIGS_DIR / "ROC_multiclass_0.pdf")
# plt.close()

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# # Compute macro-average ROC curve and ROC area
# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# # Finally average it and compute AUC
# mean_tpr /= n_classes

# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# # Plot all ROC curves
# plt.figure()
# plt.plot(
#     fpr["micro"],
#     tpr["micro"],
#     label="micro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["micro"]),
#     color="deeppink",
#     linestyle=":",
#     linewidth=4,
# )

# plt.plot(
#     fpr["macro"],
#     tpr["macro"],
#     label="macro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["macro"]),
#     color="navy",
#     linestyle=":",
#     linewidth=4,
# )
# for i, color in zip(range(n_classes), colors):
#     plt.plot(
#         fpr[i],
#         tpr[i],
#         color=color,
#         lw=lw,
#         label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
#     )

# plt.plot([0, 1], [0, 1], "k--", lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("")
# plt.legend(loc="lower right")
# plt.savefig(config.METRIC_FIGS_DIR / "ROC_multiclass_1.pdf")
# plt.close()


# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # For each class
# precision = dict()
# recall = dict()
# average_precision = dict()
# for i in range(n_classes):
#     precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
#     average_precision[i] = average_precision_score(y_bin[:, i], y_score[:, i])

# # A "micro-average": quantifying score on all classes jointly
# precision["micro"], recall["micro"], _ = precision_recall_curve(
#     y_bin.ravel(), y_score.ravel()
# )
# average_precision["micro"] = average_precision_score(y_bin, y_score, average="micro")
# print(
#     "Average precision score, micro-averaged over all classes: {0:0.2f}".format(
#         average_precision["micro"]
#     )
# )


# # setup plot details
# colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
# # random_colors = random.sample(beau_colors, 20)
# colors = cycle(colorz['hex'])

# plt.figure(figsize=(7, 8))
# f_scores = np.linspace(0.2, 0.8, num=4)
# lines = []
# labels = []
# for f_score in f_scores:
#     x = np.linspace(0.01, 1)
#     y = f_score * x / (2 * x - f_score)
#     l, = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
#     plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

# lines.append(l)
# labels.append("iso-f1 curves")
# l, = plt.plot(recall["micro"], precision["micro"], color="gold", lw=2)
# lines.append(l)
# labels.append(
#     "micro-average Precision-recall (area = {0:0.2f})"
#     "".format(average_precision["micro"])
# )

# for i, color in zip(range(n_classes), colors):
#     l, = plt.plot(recall[i], precision[i], color=color, lw=2)
#     lines.append(l)
#     labels.append(
#         "Precision-recall for class {0} (area = {1:0.2f})"
#         "".format(i, average_precision[i])
#     )

# fig = plt.gcf()
# fig.subplots_adjust(bottom=0.25)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Extension of Precision-Recall curve to multi-class")
# plt.legend(lines, labels, loc=(0, -0.38), prop=dict(size=14))


# plt.show()


#####################################################################
# Binarize the output

# factor = pd.factorize(data['diagnosis'])
# data.diagnosis = factor[0]
# definitions = factor[1]
# print(data.diagnosis.head())
# print(definitions)


# skf.get_n_splits(X, y)
# print(skf)

# for train_index, test_index in skf.split(X, y):
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# clf = lgb.LGBMClassifier(
#     # boosting_type=params["boosting_type"],
#     # colsample_bytree=params["colsample_bytree"],
#     # is_unbalance=params["is_unbalance"],
#     # learning_rate=params["learning_rate"],
#     # max_depth=params["max_depth"],
#     # min_child_samples=params["min_child_samples"],
#     # min_child_weight=params["min_child_weight"],
#     # min_split_gain=params["min_split_gain"],
#     # n_estimators=params["n_estimators"],
#     # n_jobs=params["n_jobs"],
#     # num_leaves=params["num_leaves"],
#     # num_rounds=params["num_rounds"],
#     objective=params["objective"],
#     predict_contrib=params["predict_contrib"],
#     random_state=params["random_state"],
#     # reg_alpha=params["reg_alpha"],
#     # reg_lambda=params["reg_lambda"],
#     # silent=params["silent"],
#     # subsample_for_bin=params["subsample_for_bin"],
#     # subsample_freq=params["subsample_freq"],
#     # subsample=params["subsample"],
#     verbose=params["verbose"],
# )

#Make our customer score
# def classification_report_with_accuracy_score(y_true, y_pred):
    # originalclass.extend(y_true)
    # predictedclass.extend(y_pred)
    # return accuracy_score(y_true, y_pred) # return accuracy score

# inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
# outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

# Non_nested parameter search and scoring
# clf = GridSearchCV(estimator=svr, param_grid=p_grid, cv=inner_cv)

# Nested CV with parameter optimization
# nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring=make_scorer(classification_report_with_accuracy_score))

# Average values in classification report for all folds in a K-fold Cross-validation  
# print(classification_report(originalclass, predictedclass)) 

# scores = []
# for k, (train, test) in enumerate(kf):
#     X_train, X_test = X.iloc[train], X.iloc[test]
#     Y_train, Y_test = y.iloc[train], y.iloc[test]
#     clf.fit(X_train, Y_train)
#     scoreK = clf.score(X_train, Y_train)
#     scores.append(scoreK)

#     print("Fold: {:2d}, Acc: {:.3f}".format(k + 1, scoreK))

# print("\nCV accuracy: {:.3f} +/- {:.3f}".format(np.mean(scores), np.std(scores)))

# # Passing the entirety of X and y, not X_train or y_train
# scores = cross_val_score(
#     clf, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# )
# print("Cross-validated scores:", scores)
# print("CV accuracy: {:.3f}  +/- {:.3f} ".format(np.mean(scores), np.std(scores)))
