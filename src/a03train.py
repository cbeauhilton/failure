import os
import traceback
import warnings
from itertools import cycle

import h5py
import jsonpickle
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn import datasets, svm
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import auc, classification_report, roc_curve
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

from cbh import config

print("Loading", os.path.basename(__file__))

# When training starts, certain metrics are often zero for a while, which throws a warning and clutters the terminal output
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)


data = pd.read_hdf(config.RAW_DATA_FILE_H5, key="data")
# print(list(data))

y = data["diagnosis"].copy()
X = data.drop(["diagnosis", "id"], axis=1).copy()

# print(X.head())
# print(y.head())
skf = StratifiedKFold(n_splits=5, random_state=config.SEED)
skf.get_n_splits(X, y)

# print(skf)

for train_index, test_index in skf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# print(y.value_counts())
# Binarize the output
y_bin = label_binarize(y, classes=["mds", "cmml", "pmf", "et", "pv"])
n_classes = y_bin.shape[1]


params = config.PARAMS_LGBM
early_stopping_rounds = 1000
# gbm_model = lgb.LGBMClassifier(**params) # <-- could also do this, but it's kind of nice to have it all explicit
clf = lgb.LGBMClassifier(
    # boosting_type=params["boosting_type"],
    # colsample_bytree=params["colsample_bytree"],
    # is_unbalance=params["is_unbalance"],
    # learning_rate=params["learning_rate"],
    # max_depth=params["max_depth"],
    # min_child_samples=params["min_child_samples"],
    # min_child_weight=params["min_child_weight"],
    # min_split_gain=params["min_split_gain"],
    # n_estimators=params["n_estimators"],
    # n_jobs=params["n_jobs"],
    # num_leaves=params["num_leaves"],
    # num_rounds=params["num_rounds"],
    objective=params["objective"],
    predict_contrib=params["predict_contrib"],
    random_state=params["random_state"],
    # reg_alpha=params["reg_alpha"],
    # reg_lambda=params["reg_lambda"],
    # silent=params["silent"],
    # subsample_for_bin=params["subsample_for_bin"],
    # subsample_freq=params["subsample_freq"],
    # subsample=params["subsample"],
    verbose=params["verbose"],
)

clf.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    # eval_metric="logloss",
    early_stopping_rounds=early_stopping_rounds,
)


def lgbm_save_model_to_h5():
    """
    To retrieve pickled model, use something like:
    `pkl_model = metricsgen.save_model_to_pickle()`

    then: 

    `with open(pkl_model, "rb") as fin:
            gbm_model = pickle.load(fin)` 
    """

    print("Dumping model with pickle...")
    print("JSONpickling the model...")
    frozen = jsonpickle.encode(clf)
    print("Saving clf to .h5 file...")
    h5_file = "test.h5"
    with h5py.File(h5_file, 'a') as f:
        try:
            f.create_dataset('clf', data=frozen)
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
            try:
                del f["clf"]
                f.create_dataset('clf', data=frozen)
                print("Successfully deleted old clf and saved new one!")
            except:
                print("Old clf persists...")
    print(h5_file)

lgbm_save_model_to_h5()


def lgbm_save_feature_importance_plot():
    print("Plotting feature importances...")
    
    ax = lgb.plot_importance(
        clf, figsize=(5, 20), importance_type="gain", precision=2
    )
    plt.savefig((config.METRIC_FIGS_DIR/"feature_importance.pdf"), dpi=1200, transparent=False, bbox_inches="tight" )
    plt.close()

lgbm_save_feature_importance_plot()


print(classification_report(y, clf.predict(X)))

y_score = cross_val_predict(clf, X, y, cv=10, method="predict_proba")

fpr = dict()
tpr = dict()
roc_auc = dict()
lw = 2
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(["blue", "red", "green", "aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label=f"ROC curve of class {i} (area = {roc_auc[i]:0.2f})",
    )
plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("")
plt.legend(loc="lower right")
plt.savefig(config.METRIC_FIGS_DIR/"multiclassROC0.pdf")
plt.close()

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("")
plt.legend(loc="lower right")
plt.savefig(config.METRIC_FIGS_DIR/"multiclassROC1.pdf")
plt.close()
