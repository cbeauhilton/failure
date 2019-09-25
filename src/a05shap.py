import os
import pickle
import random
import shutil
import time
import warnings
from itertools import cycle

import h5py
import jsonpickle
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split

from cbh import config
from cbh.dumb import get_xkcd_colors
from cbh.exhandler import exhandler
from cbh.fancyimpute import SoftImputeDf

colorz = get_xkcd_colors()
colors = cycle(colorz["hex"])

warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore")
print("Loading", os.path.basename(__file__))

# Load data
data = pd.read_hdf(config.RAW_DATA_FILE_H5, key="data")
y = pd.read_hdf(config.RAW_DATA_FILE_H5, key="y")
X = pd.read_hdf(config.RAW_DATA_FILE_H5, key="X")
X = X.round(2)

classes = np.unique(y) # retrieve all class names
CLASSES = [x.upper() for x in classes] # make uppercase version
# print(CLASSES)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=config.SEED
)


# Define classifier
params = config.PARAMS_LGBM
clf = lgb.LGBMClassifier(**params)
early_stopping_rounds = 500
cats = list(X.select_dtypes(include='category'))
# print(cats)
X_train = SoftImputeDf().fit_transform(X_train)
X_train[cats] = X_train[cats].astype('category')
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
shap_values = explainer.shap_values(features_shap)
shap_expected = explainer.expected_value


print("Dumping model pickle...")
save_model_file = config.MODEL_PICKLE
pickle.dump(clf, open(save_model_file, "wb"))

print("Dumping SHAP to h5...")

f = h5py.File(config.MODEL_SHAP_H5, 'w', libver='latest')
dset = f.create_dataset('shap_values', data=shap_values, compression='gzip', compression_opts=9)

dset.attrs['Description'] = '''
SHAP values as a numpy array. 
'''

d = {"shap_expected": shap_expected}
for k, v in d.items():
    f.create_dataset(str(k), data=v)

f.close()

f = h5py.File(config.MODEL_SHAP_H5, 'a', libver='latest')
f.attrs["shap_expected"] = shap_expected
f.attrs["classes"] = list(classes)
f.close()
# h5_classes = np.array(classes)
# dt = h5py.string_dtype(encoding='ascii')
# f.create_dataset("classes", data=h5_classes, dtype=dt)



# def shap_save_to_disk(h5_file, target, shap_expected, shap_values, features_shap):
#     print("Saving SHAP to .h5 file...")
#     shap_val_df = pd.DataFrame(shap_values)
#     shap_feat_df = pd.DataFrame(features_shap)
#     # define what goes in the first row with "d"
#     d = [[target, shap_expected]]
#     exp_df = pd.DataFrame(
#         d, columns=("target", "shap_exp_val")
#     )
#     shap_val_df.to_hdf(h5_file, key="shap_values")
#     shap_feat_df.to_hdf(h5_file, key="features_shap")
#     exp_df.to_hdf(h5_file, key="shap_expected_value")

# shap_save_to_disk(h5_file=config.MODELS_DIR/"shap.h5", target="diagnosis", shap_expected=shap_expected, shap_values=shap_values, features_shap=features_shap)

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
right_pred = combo[combo["Success"]==True]
right_pt_nums = list(right_pred["Sample_Number"])
wrong_pt_nums = list(wrong_pred["Sample_Number"])
# grab a random sample of 10 from each
pt_nums_right = random.sample(right_pt_nums, 20)
pt_nums_wrong = random.sample(wrong_pt_nums, 20)

force_plot_csv = pd.DataFrame()
force_plot_csv['correctly_classified'] = sorted(pt_nums_right)
force_plot_csv['incorrectly_classified'] = sorted(pt_nums_wrong)
force_plot_csv.to_csv(config.TABLES_DIR / "pt_nums.csv")
# print(force_plot_csv)
pt_nums = list(pt_nums_right) + list(pt_nums_wrong)
# print(pt_nums)


print("Saving SHAP values to disk in order of importance...")

imp_cols = pd.DataFrame()
for i, classname in enumerate(classes):
    shaps = pd.DataFrame(shap_values[i], columns=features_shap.columns.values)
    shaps = shaps.abs().mean().sort_values(ascending=False).index.tolist()
    shaps = pd.DataFrame(shaps, columns=[classes[i].upper()])
    imp_cols = pd.concat([imp_cols, shaps], axis=1)

# tops = pd.DataFrame((np.abs(shap_values).mean(0)))
# # tops = pd.DataFrame((np.abs(shap_values)))
# class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
# tops.columns = features_shap.columns.values
# print(tops.head())
# shaps = pd.DataFrame(shap_values, columns=features_shap.columns.values)
# shaps = pd.DataFrame((np.abs(shap_values).mean(0).sort_values(ascending=False).index.tolist()), columns=features_shap.columns.values)
# shaps = pd.DataFrame(shaps, columns="All")
# imp_cols = pd.concat([imp_cols, shaps], axis=1)


print(imp_cols.head())

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
        time.sleep(3)
    except OSError as e:
        # pass
        print ("%s - %s." % (e.filename, e.strerror))
        print(f"Attempting to make {e.filename}...")
    if not os.path.exists(config.FIGURES_DIR/f"shap_images/{folder}"):
        os.makedirs(config.FIGURES_DIR/f"shap_images/{folder}") 
        print(f"Made directory: {folder}")


prettycols = pd.read_csv(config.TABLES_DIR / "prettify.csv")
di = dict(zip(prettycols.ugly, prettycols.pretty_full))
di2 = dict(zip(prettycols.ugly, prettycols.pretty_abbr))
feature_names = X.columns.map(di)
feature_names_short = X.columns.map(di2)

save_feature_names_short = feature_names_short.to_list()
save_feature_names = feature_names.to_list()
f = h5py.File(config.MODEL_SHAP_H5, 'a', libver='latest')

f.attrs["feature_names_short"] = save_feature_names_short
f.attrs["feature_names"] = save_feature_names
f.close()

print(f"SHAP summary bar...")
shap.summary_plot(shap_values, X, plot_type="bar", class_names = CLASSES, feature_names=feature_names, show=False)
# plt.title(f"{classname.upper()}")
plt.savefig(
    config.FIGURES_DIR / "shap_images/shap_summary"/ f"shap_summary_bar_all_classes.pdf", bbox_inches="tight"
)
plt.close()

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
            exhandler(ex, module=os.path.basename(__file__))

    try:
        print(f"{classname} compact dot summary...")
        shap.summary_plot(shap_values[i], X, show=False, plot_type="compact_dot")
        plt.title(f"{classname.upper()}")
        plt.savefig(
            config.FIGURES_DIR / f"shap_images/shap_summary/shap_summary_compact_dot_{classname}_{i}.pdf",
            bbox_inches="tight",
        )
        plt.close()

    except Exception as ex:
            exhandler(ex, module=os.path.basename(__file__))

    print(f"Dependence plots for {classname}...")
    for col in list(X):
        try:
            # print(f"Dependence plot for {col} for {classname}...")
            shap.dependence_plot(f"{col}", shap_values[i], X, show=False)
            plt.tight_layout()
            if not os.path.exists(config.FIGURES_DIR / f"shap_images/shap_dependence/{classname}"):
                os.makedirs(config.FIGURES_DIR / f"shap_images/shap_dependence/{classname}") 
            plt.savefig(config.FIGURES_DIR / f"shap_images/shap_dependence/{classname}/shap_dependence_{classname}_{col}.pdf", transparent=True)
            plt.close()
        except Exception as ex:
            exhandler(ex, module=os.path.basename(__file__))
        
    print(f"{classname} force plots...")
    for pt_num in pt_nums:
        try:
            try:
                blah = X_test.loc[X_test['pt_num'] == pt_num]
                if blah is not None:
                    pass
                    # print(f"{pt_num} works")
            except:
                print(f"Exception at pt_num: {pt_num}")
            print(X_test.loc[X_test['pt_num'] == pt_num])
            shap.force_plot(
                explainer.expected_value[i],
                shap_values[i][pt_num, :],
                X_test.loc[X_test['pt_num'] == pt_num],
                # X_test.iloc[f"{pt_num}", :],
                link="logit",
                matplotlib=True,
                feature_names=feature_names_short,
                show=False,
            )
            plt.title(f"{classname.upper()}", fontsize='xx-large', backgroundcolor= 'white', y=1.08)
            plt.savefig(
                config.FIGURES_DIR
                / f"shap_images/force_plots/forceplot_{classname}_{i}_pt_{pt_num}.pdf",
                bbox_inches="tight",
                transparent=True,
            )
            plt.close()
        except Exception as ex:
            exhandler(ex, module=os.path.basename(__file__))


imp_cols = pd.read_csv(config.TABLES_DIR / "shap_df.csv")
prettycols = pd.read_csv(config.TABLES_DIR / "prettify.csv")
di = dict(zip(prettycols.ugly, prettycols.pretty_full))
print(di)
pretty_imp_cols = pd.DataFrame()
for classname in CLASSES:
    pretty_imp_cols[f'{classname}'] = imp_cols[f'{classname}'].map(di).fillna(imp_cols[f'{classname}'])
# print(pretty_imp_cols.head())

# features_shap.columns = features_shap.columns.to_series().map(
#             di["feature_pretty"]
#         )

pretty_imp_cols.to_csv(config.TABLES_DIR / "shap_df_pretty.csv")

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
