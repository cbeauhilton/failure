import csv
import json
import os
import random
import traceback
import warnings
from inspect import signature
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
from cbh.colors_rando import *
from cbh.dumb import get_xkcd_colors

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size

colorz = get_xkcd_colors()
xkcd_cycle = cycle(colorz['hex'])
colorblind9 = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
tableau10 = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
colorbrewer8 = ["#d73027", "#f46d43", "#fdae61", "#fee090", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4",]

# colors = tableau10
colors = colorbrewer8
# colors = colorblind9

print("Loading", os.path.basename(__file__))

# Load data
data = pd.read_hdf(config.RAW_DATA_FILE_H5, key="data")
y = pd.read_hdf(config.RAW_DATA_FILE_H5, key="y")
X = pd.read_hdf(config.RAW_DATA_FILE_H5, key="X")
# print(list(data))

metric_image_folders = ["AUC", "PR", "calibration"]
for folder in metric_image_folders:
    if not os.path.exists(config.METRIC_FIGS_DIR/f"{folder}"):
        os.makedirs(config.METRIC_FIGS_DIR/f"{folder}") 
        # print(f"Made directory: {folder}")

# Define labels and features, and binarize labels for AUC/PR curves
# print(X.head())
# print(y.head())
classes = np.unique(y)
# print(classes)
y_true = label_binarize(y, classes=classes)
n_classes = y_true.shape[1]

# Define cross-validation scheme
n_splits = 10


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
f_scores = np.linspace(0.2, 0.8, num=4)
# print(mean_fpr)
misclassified_df = pd.read_csv(config.METRIC_FIGS_DIR / "misclassified_cv.csv")

roc_data = pd.read_csv(config.METRIC_FIGS_DIR / "roc_curve_data.csv")
roc_data = roc_data.replace("nan", np.nan)
# print(roc_data.head(100))


with open(config.TABLES_DIR/'perf_dict.json', 'r') as f:
    perf_dict = json.load(f)

# report[i][classes[j]]["brier_score"] = brier_score.round(10)
# report[i][classes[j]]["roc_auc"] = roc_auc[j].round(10)
# report[i][classes[j]]["z_avg_precision"] = average_precision[j]


for j in range(len(classes)):
    # print(j)
    roc_aucs = []
    brier_scores = []
    avg_precisions = []
    precisions = []
    recalls = []
    perf_dict[classes[j]] = {}
    for i in range(n_splits): 
        # print(i) 
        # print(classes[j])
        roc_auc = perf_dict[f"{i}"][classes[j]]["roc_auc"]
        roc_aucs.append(roc_auc)
        avg_ps = perf_dict[f"{i}"][classes[j]]["z_avg_precision"]
        avg_precisions.append(avg_ps)
        b_s = perf_dict[f"{i}"][classes[j]]["brier_score"]
        brier_scores.append(b_s)
        precs = perf_dict[f"{i}"]["z_classification_reports"][classes[j]]["precision"]
        precisions.append(precs)
        recs = perf_dict[f"{i}"]["z_classification_reports"][classes[j]]["recall"]
        recalls.append(recs)
    # print(roc_aucs)
    # print(precisions)
    # print(recalls)
    mean_auc = np.mean(roc_aucs)
    std_auc = np.std(roc_aucs)
    perf_dict[classes[j]]["mean_auc"] = mean_auc
    perf_dict[classes[j]]["std_auc"] = std_auc
    mean_brier = np.mean(brier_scores)
    std_brier = np.std(brier_scores)
    perf_dict[classes[j]]["mean_brier"] = mean_brier
    perf_dict[classes[j]]["std_brier"] = std_brier
    mean_pa = np.mean(avg_precisions)
    std_pa = np.std(avg_precisions)
    perf_dict[classes[j]]["mean_pa"] = mean_pa
    perf_dict[classes[j]]["std_pa"] = std_pa
    mean_precs = np.mean(precisions)
    std_precs = np.std(precisions)
    perf_dict[classes[j]]["mean_precisions"] = mean_precs
    perf_dict[classes[j]]["std_precisions"] = std_precs
    mean_recs = np.mean(recalls)
    std_recs = np.std(recalls)
    perf_dict[classes[j]]["mean_recalls"] = mean_recs
    perf_dict[classes[j]]["std_recalls"] = std_recs
    # print(mean_pa, std_pa, mean_auc, std_auc, mean_brier, std_brier)

for classname in classes:
    # select the columns with the appropriate classname
    df2 = roc_data.filter(regex=classname)
    fprs = df2.filter(regex="fpr")
    tprss = df2.filter(regex="tpr")
    aucss = df2.filter(regex="auc")
    for i, fold_num in enumerate(range(n_classes)):
        fpr = fprs.filter(regex=f"{fold_num}_{classname}")
        fpr = fpr.dropna()
        fpr = fpr[fpr.columns[0:3]].values.T
        fpr = fpr[0, :]
        # fpr = np.reshape(fpr, len(fpr))
        # print(type(fpr))
        # print(fpr)

        tpr = tprss.filter(regex=f"{fold_num}_{classname}")
        tpr = tpr.dropna()
        tpr = tpr[tpr.columns[0:3]].values.T
        tpr = tpr[0, :]

        roc_auc = aucss.filter(regex=f"{fold_num}_{classname}")
        roc_auc = roc_auc.iloc[0][0]
        # print(type(roc_auc))

        tprs.append(interp(mean_fpr, fpr, tpr))
        # print(type(tprs))
        tprs[-1][0] = 0.0

        aucs.append(roc_auc)
        plt.plot(
            fpr, tpr, lw=1, alpha=0.3, #label="ROC fold %d (AUC = %0.2f)" % (i, roc_auc)
        )

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    # print("\n")
    # print(type(tprs))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    # perf_dict[f"{classname}"] = {}
    # perf_dict[f"{classname}"]['auc'] = mean_auc
    # perf_dict[f"{classname}"]['auc_std'] = std_auc
 
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("")
    plt.legend(loc="lower right")
    plt.savefig(
        (config.METRIC_FIGS_DIR / f"AUC/AUC_ROC_{classname}.pdf"),
        dpi=1200,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()

    for i, fold_num in enumerate(range(n_splits)):
        ### todo: recall_micro for all of n_splits in one graph

        aps = []
        
        for trial in range(n_splits):
            average_precision = perf_dict[f"{trial}"][classname]["z_avg_precision"]
            aps.append(average_precision)
        ap_std = np.std(aps)
        average_precision = perf_dict[f"{fold_num}"][classname]["z_avg_precision"]
        brier_score = perf_dict[f"{fold_num}"][classname]["brier_score"]
        labels = perf_dict[f"{fold_num}"][classname]["z_labels"]
        # preds = np.array(perf_dict[f"{fold_num}"][classname]["z_preds"])
        scores = np.array(perf_dict[f"{fold_num}"][classname]["z_scores"])
        precision = np.array(perf_dict[f"{fold_num}"][classname]["z_precision"])
        recall = np.array(perf_dict[f"{fold_num}"][classname]["z_recall"])
        average_precision = perf_dict[f"{fold_num}"][classname]["z_avg_precision"]
        classification_report = perf_dict[f"{fold_num}"]["z_classification_reports"]


        step_kwargs = (
            {"step": "post"} if "step" in signature(plt.fill_between).parameters else {}
        )
        # plt.title(f"Precision-Recall Curve")
        plt.step(recall, precision, color="b", alpha=0.3, where="post")
        plt.fill_between(recall, precision, alpha=0.1, color="b", **step_kwargs)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend([f"Average Precision: {average_precision:0.2f} $\pm$ {ap_std:0.2f} "],handletextpad=0, handlelength=0, loc="lower right")
        figure_title = (
            f"{classname}_Precision_Recall_curve_AP_{average_precision*100:.0f}_"
        )
    plt.savefig(
        (config.METRIC_FIGS_DIR / f"PR/PR_combo_{classname}.pdf"),
        dpi=1200,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()

    for i, fold_num in enumerate(range(n_splits)):
        labels = perf_dict[f"{fold_num}"][classname]["z_labels"]
        scores = np.array(perf_dict[f"{fold_num}"][classname]["z_scores"])
        brier_score = perf_dict[f"{fold_num}"][classname]["brier_score"]
        gb_y, gb_x = calibration_curve(labels, scores, n_bins=100)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(gb_x, gb_y, marker=".", color="red")
        plt.xlabel("Predicted probability")
        plt.ylabel("True probability")
        plt.legend([f"Brier Score Loss: {brier_score:.2f}"], handletextpad=0, handlelength=0,  loc="lower right")
        plt.savefig(
            (config.METRIC_FIGS_DIR / f"calibration/calibration_curve_{classname}.pdf"),
            dpi=1200,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close()

#     plt.show()

# print(perf_dict)

with open(config.TABLES_DIR/'perf_dict.json', 'w') as f:
    json.dump(perf_dict, f)

# plt.figure(figsize=(10, 10))
# fig.set_size_inches(18.5, 10.5)
for classname, color in zip(classes, colors):
    # select the columns with the appropriate classname
    df2 = roc_data.filter(regex=classname)
    fprs = df2.filter(regex="fpr")
    tprss = df2.filter(regex="tpr")
    aucss = df2.filter(regex="auc")
    for i, fold_num in enumerate(range(n_classes)):
        fpr = fprs.filter(regex=f"{fold_num}_{classname}")
        fpr = fpr.dropna()
        fpr = fpr[fpr.columns[0:3]].values.T
        fpr = fpr[0, :]
        # fpr = np.reshape(fpr, len(fpr))
        # print(type(fpr))
        # print(fpr)

        tpr = tprss.filter(regex=f"{fold_num}_{classname}")
        tpr = tpr.dropna()
        tpr = tpr[tpr.columns[0:3]].values.T
        tpr = tpr[0, :]

        roc_auc = aucss.filter(regex=f"{fold_num}_{classname}")
        roc_auc = roc_auc.iloc[0][0]
        # print(type(roc_auc))

        tprs.append(interp(mean_fpr, fpr, tpr))
        # print(type(tprs))
        tprs[-1][0] = 0.0

        aucs.append(roc_auc)
        # plt.plot(
        #     fpr, tpr, lw=1, alpha=0.3, #label="ROC fold %d (AUC = %0.2f)" % (i, roc_auc)
        # )

    # print("\n")
    # print(type(tprs))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    # perf_dict[f"{classname}"] = {}
    # perf_dict[f"{classname}"]['auc'] = mean_auc
    # perf_dict[f"{classname}"]['auc_std'] = std_auc
    if classname == "mds_mpn":
        classname = "MDS/MPN"
    plt.plot(
        mean_fpr,
        mean_tpr,
        color=color,
        label=f"{classname.upper()} AUC {mean_auc:0.2f} $\pm$ {std_auc:0.2f}",
        lw=0.75,
        alpha=0.8,
    )
plt.plot([0, 1], [0, 1], linestyle="--", lw=0.5, color="r", label="Chance", alpha=0.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("")
plt.legend(loc="lower right")
plt.savefig(
    (config.METRIC_FIGS_DIR / f"AUC/AUC_ROC_combo.pdf"),
    dpi=1200,
    transparent=True,
    bbox_inches="tight",
)
plt.close()


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

# # for train, test in k.split(X, y):
# #     # y.iloc[test] = label_binarize(y.iloc[test], classes =["mds", "cmml", "pmf", "et", "pv"])
# #     probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])

# #     # Compute ROC curve and ROC area for each class
# #     fpr = dict()
# #     tpr = dict()
# #     roc_auc = dict()
# #     for idx in range(n_classes):
# #         fpr[idx], tpr[idx], _ = roc_curve(y_test[:, idx], y_score[:, idx])
# #         roc_auc[idx] = auc(fpr[idx], tpr[idx])

# #     fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, idx])
# #     tprs.append(interp(mean_fpr[idx], fpr[idx], tpr[idx]))
# #     tprs[-1][0] = 0.0
# #     roc_auc = auc(fpr[idx], tpr[idx])
# #     aucs.append(roc_auc)
# #     plt.plot(
# #         fpr[idx],
# #         tpr[idx],
# #         lw=1,
# #         alpha=0.3,
# #         label=f"ROC fold %d (AUC = %0.2f)" % (i, roc_auc),
# #     )

# #     i += 1
# # plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

# # mean_tpr = np.mean(tprs, axis=0)
# # mean_tpr[-1] = 1.0
# # mean_auc = auc(mean_fpr, mean_tpr)
# # std_auc = np.std(aucs)
# # plt.plot(
# #     mean_fpr,
# #     mean_tpr,
# #     color="b",
# #     label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
# #     lw=2,
# #     alpha=0.8,
# # )

# # std_tpr = np.std(tprs, axis=0)
# # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# # plt.fill_between(
# #     mean_fpr,
# #     tprs_lower,
# #     tprs_upper,
# #     color="grey",
# #     alpha=0.2,
# #     label=r"$\pm$ 1 std. dev.",
# # )

# # plt.xlim([-0.05, 1.05])
# # plt.ylim([-0.05, 1.05])
# # plt.xlabel("False Positive Rate")
# # plt.ylabel("True Positive Rate")
# # plt.title("Receiver operating characteristic example")
# # plt.legend(loc="lower right")
# # plt.show()


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
# plt.savefig(config.METRIC_FIGS_DIR / "multiclassROC0.pdf")
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
# plt.savefig(config.METRIC_FIGS_DIR / "multiclassROC1.pdf")
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


# #####################################################################
# # Binarize the output

# # factor = pd.factorize(data['diagnosis'])
# # data.diagnosis = factor[0]
# # definitions = factor[1]
# # print(data.diagnosis.head())
# # print(definitions)


# # skf.get_n_splits(X, y)
# # print(skf)

# # for train_index, test_index in skf.split(X, y):
# #     # print("TRAIN:", train_index, "TEST:", test_index)
# #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
# #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# # clf = lgb.LGBMClassifier(
# #     # boosting_type=params["boosting_type"],
# #     # colsample_bytree=params["colsample_bytree"],
# #     # is_unbalance=params["is_unbalance"],
# #     # learning_rate=params["learning_rate"],
# #     # max_depth=params["max_depth"],
# #     # min_child_samples=params["min_child_samples"],
# #     # min_child_weight=params["min_child_weight"],
# #     # min_split_gain=params["min_split_gain"],
# #     # n_estimators=params["n_estimators"],
# #     # n_jobs=params["n_jobs"],
# #     # num_leaves=params["num_leaves"],
# #     # num_rounds=params["num_rounds"],
# #     objective=params["objective"],
# #     predict_contrib=params["predict_contrib"],
# #     random_state=params["random_state"],
# #     # reg_alpha=params["reg_alpha"],
# #     # reg_lambda=params["reg_lambda"],
# #     # silent=params["silent"],
# #     # subsample_for_bin=params["subsample_for_bin"],
# #     # subsample_freq=params["subsample_freq"],
# #     # subsample=params["subsample"],
# #     verbose=params["verbose"],
# # )

# #Make our customer score
# # def classification_report_with_accuracy_score(y_true, y_pred):
#     # originalclass.extend(y_true)
#     # predictedclass.extend(y_pred)
#     # return accuracy_score(y_true, y_pred) # return accuracy score

# # inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
# # outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

# # Non_nested parameter search and scoring
# # clf = GridSearchCV(estimator=svr, param_grid=p_grid, cv=inner_cv)

# # Nested CV with parameter optimization
# # nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring=make_scorer(classification_report_with_accuracy_score))

# # Average values in classification perf_dict for all folds in a K-fold Cross-validation
# # print(classification_report(originalclass, predictedclass))

# # scores = []
# # for k, (train, test) in enumerate(kf):
# #     X_train, X_test = X.iloc[train], X.iloc[test]
# #     Y_train, Y_test = y.iloc[train], y.iloc[test]
# #     clf.fit(X_train, Y_train)
# #     scoreK = clf.score(X_train, Y_train)
# #     scores.append(scoreK)

# #     print("Fold: {:2d}, Acc: {:.3f}".format(k + 1, scoreK))

# # print("\nCV accuracy: {:.3f} +/- {:.3f}".format(np.mean(scores), np.std(scores)))

# # # Passing the entirety of X and y, not X_train or y_train
# # scores = cross_val_score(
# #     clf, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# # )
# # print("Cross-validated scores:", scores)
# # print("CV accuracy: {:.3f}  +/- {:.3f} ".format(np.mean(scores), np.std(scores)))
