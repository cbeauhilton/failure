import csv
import os
import random
import shutil
import sys
import time
import traceback
import warnings
from itertools import cycle
from pathlib import Path

import h5py
import jsonpickle
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from PyPDF2 import PdfFileReader, PdfFileWriter
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
from cbh.exhandler import exhandler

colorz = get_xkcd_colors()
colors = cycle(colorz["hex"])

warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore")
print("Loading", os.path.basename(__file__))




# Load data
data = pd.read_hdf(config.RAW_DATA_FILE_H5, key="data")
data = data.reset_index(drop=True)
data = data.round(2)

# data['id'] = data['id'].astype('category')
data['pt_num'] = data.index
data['pt_num'] = data['pt_num'].astype('category')
# print(data.id)
# print(data.tail(30))
# print(list(data))

# Define labels and features, and binarize labels for AUC/PR curves
y = data["diagnosis"].copy()
X = data.drop(["diagnosis", "id"], axis=1).copy()
# X = data.drop(["diagnosis"], axis=1).copy()
classes = np.unique(y) # retrieve all class names
CLASSES = [x.upper() for x in classes] # make uppercase version
# print(CLASSES)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
# print(X_test.head())

# Define classifier
params = config.PARAMS_LGBM
clf = lgb.LGBMClassifier(**params)
early_stopping_rounds = 500
model = clf.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    # eval_metric="logloss",
    early_stopping_rounds=early_stopping_rounds,
    verbose=False
)
evals_result = model._evals_result
explainer = shap.TreeExplainer(clf)
features_shap = X.copy()
# features_shap = X.sample(n=len(X), random_state=config.SEED, replace=False)
shap_values = explainer.shap_values(features_shap)
# print(shap_values)



y_score = model.predict_proba(X_test)
# print(y_score)
y_pred = model.predict(X_test)

# Generate dataframes
# Using a cutoff
actual = y_test
successes = actual == y_pred
predictions = pd.DataFrame(
    {
        "Success": successes,
        "Prediction": y_pred,
        "Actual": actual,
        "Sample_Number": actual.index,
    }
)
predictions = predictions.reset_index(drop=True)

# Using raw probabilities
probs = pd.DataFrame(y_score, columns=[x.upper() for x in classes])
probs = probs.reset_index(drop=True)

# Combine dataframes
combo = pd.concat([predictions, probs], axis=1)
combo = combo.reset_index(drop=True)
# misclassified_df = pd.DataFrame()
# misclassified_df = misclassified_df.append(combo)
# misclassified_df = misclassified_df.reset_index(drop=True)

# Save combined dataframes
combo.to_csv(config.TABLES_DIR / "misclassified.csv")



wrong_pred = combo[combo["Success"]==False]
# print(wrong_pred)
right_pred = combo[combo["Success"]==True]
right_pt_nums = list(right_pred["Sample_Number"])
wrong_pt_nums = list(wrong_pred["Sample_Number"])


pt_nums_right = random.sample(right_pt_nums, 10)
pt_nums_wrong = random.sample(wrong_pt_nums, 10)

force_plot_csv = pd.DataFrame()
force_plot_csv['correctly_classified'] = sorted(pt_nums_right)
force_plot_csv['incorrectly_classified'] = sorted(pt_nums_wrong)
force_plot_csv.to_csv(config.TABLES_DIR / "pt_nums.csv")
# print(force_plot_csv)


pt_nums = pt_nums_right + pt_nums_wrong
# print(pt_nums)


print("Saving SHAP values to disk in order of importance...")



imp_cols = pd.DataFrame()
for i, classname in enumerate(classes):
    shaps = pd.DataFrame(shap_values[i], columns=features_shap.columns.values)
    shaps = shaps.abs().mean().sort_values(ascending=False).index.tolist()
    shaps = pd.DataFrame(shaps, columns=[classes[i].upper()])
    imp_cols = pd.concat([imp_cols, shaps], axis=1)

csv_file = config.TABLES_DIR / "shap_df.csv"
h5_file = config.PROCESSED_DATA_DIR / "everything.h5"
imp_cols.to_csv(csv_file)
imp_cols.to_hdf(h5_file, key="ordered_shap_cols", format="table")

print(f"CSV available at {csv_file}")
print(f"H5 available at {h5_file}")

dirs = ["shap_summary", "shap_dependence", "force_plots"]

for folder in dirs:
    try:
        shutil.rmtree(config.FIGURES_DIR/f"shap_images/{folder}")
    except OSError as e:
        # pass
        print ("Error: %s - %s." % (e.filename, e.strerror))
    if not os.path.exists(config.FIGURES_DIR/f"shap_images/{folder}"):
        os.makedirs(config.FIGURES_DIR/f"shap_images/{folder}") 



print(f"SHAP summary bar...")
shap.summary_plot(shap_values, X, plot_type="bar", class_names = CLASSES, show=False)
# plt.title(f"{classname.upper()}")
plt.savefig(
    config.FIGURES_DIR / "shap_images/shap_summary"/ f"shap_summary_bar_all_classes.pdf", bbox_inches="tight"
)
plt.close()

n_force_plots = config.N_FORCE_PLOTS

for i, classname in enumerate(CLASSES):
    try:
        print(f"{classname} dot summary...")
        shap.summary_plot(shap_values[i], X, show=False)
        plt.title(f"{classname.upper()}")
        plt.savefig(
            config.FIGURES_DIR / f"shap_images/shap_summary/shap_summary_dot_{classname}_{i}.pdf",
            bbox_inches="tight",
        )
        plt.close()

        print(f"{classname} bar summary...")
        shap.summary_plot(shap_values[i], X, plot_type="bar", show=False)
        plt.title(f"{classname.upper()}")
        plt.savefig(
            config.FIGURES_DIR / f"shap_images/shap_summary/shap_summary_bar_{classname}_{i}.pdf",
            bbox_inches="tight",
        )
        plt.close()
    except Exception as ex:
            exhandler(ex)

    for col in list(X):
        try:
            shap.dependence_plot(f"{col}", shap_values[i], X, show=False)
            plt.tight_layout()
            plt.savefig(config.FIGURES_DIR / f"shap_images/shap_dependence/shap_dependence_{col}.pdf", transparent=True)
            plt.close()
        except Exception as ex:
            exhandler(ex)
        
    print(f"{classname} force plots...")

    for pt_num in pt_nums:
        try:
            shap.force_plot(
                explainer.expected_value[i],
                shap_values[i][pt_num, :],
                X_test.iloc[pt_num, :],
                link="logit",
                matplotlib=True,
                show=False,
            )
            plt.title(f"{classname.upper()}")
            plt.savefig(
                config.FIGURES_DIR
                / "shap_images/force_plots"
                / f"forceplot_{classname}_{i}_pt_{pt_num}.pdf",
                bbox_inches="tight",
                transparent=True,
            )
            plt.close()
        except Exception as ex:
            exhandler(ex)




timestr = time.strftime("_%Y-%m-%d-%H%M_")




print("finis")


# for i in range(len(classes)):
#     print(f"Data {len(data)}")
#     print(f"X {len(X)}")
#     print(f"y_test {len(y_test)}")
#     print(f"features_shap {len(features_shap)}")
#     print(f"shap_values {len(shap_values[i])}")
#     print(f"combo {len(combo)}")
#     print(f"probs {len(probs)}")
#     print(f"predictions {len(predictions)}")

# if not os.path.exists(config.FIGURES_DIR/"shap_images/shap_summary"):
#     os.makedirs(config.FIGURES_DIR/"shap_images/shap_summary")
# if not os.path.exists(config.FIGURES_DIR / "shap_images/shap_dependence/"):
#     os.makedirs(config.FIGURES_DIR / "shap_images/shap_dependence/")
# if not os.path.exists(config.FIGURES_DIR / "shap_images/force_plots/"):
#     os.makedirs(config.FIGURES_DIR / "shap_images/force_plots/")