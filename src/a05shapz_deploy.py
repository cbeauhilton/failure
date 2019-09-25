import os
import pickle
import warnings

import h5py
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split

from cbh import config

warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

print("Loading", os.path.basename(__file__))

# Load data
data = pd.read_hdf(config.RAW_DATA_FILE_H5, key="data")
y = pd.read_hdf(config.RAW_DATA_FILE_H5, key="y")
X = pd.read_hdf(config.RAW_DATA_FILE_H5, key="X")
drop_vus = [col for col in list(X) if "_vus" in col]
X = X.drop(columns=drop_vus)
X = X.drop(columns=['pt_num'])
X = X.round(2)

# Select top variables by their SHAP values
csv_file = config.TABLES_DIR / "shap_df.csv"
top_shaps = pd.read_csv(csv_file, nrows=31)
top_shaps = top_shaps.drop(columns="Unnamed: 0")
top_cols = []
for col in list(top_shaps):
    top_cols.append(list(top_shaps[col]))

flat_list = [item for sublist in top_cols for item in sublist]
flat_list = list(set(flat_list))
flat_list = [col for col in flat_list if "_vus" not in col]
final_list = list(set(flat_list) & set(list(X)))
final_list.sort()
X = X[final_list]


classes = np.unique(y) # retrieve all class names
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=config.SEED
)


# Define classifier
params = config.PARAMS_LGBM
clf = lgb.LGBMClassifier(**params)
early_stopping_rounds = 500
cats = list(X.select_dtypes(include='category'))

model = clf.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=early_stopping_rounds,
    verbose=False
)


evals_result = model._evals_result
explainer = shap.TreeExplainer(clf)
features_shap = X.copy()
shap_values = explainer.shap_values(features_shap)
shap_expected = explainer.expected_value

save_model_file = config.DEPLOY_MODEL_PICKLE
save_shap_file = config.DEPLOY_MODEL_SHAP_H5

print("Dumping model pickle...")
pickle.dump(model, open(save_model_file, "wb"))

print("Dumping SHAP to h5...")
f = h5py.File(save_shap_file, 'w', libver='latest')
dset = f.create_dataset('shap_values', data=shap_values, compression='gzip', compression_opts=9)

dset.attrs['Description'] = '''
SHAP values as a numpy array. 
'''

d = {"shap_expected": shap_expected}
for k, v in d.items():
    f.create_dataset(str(k), data=v)

f.close()

f = h5py.File(save_shap_file, 'a', libver='latest')
f.attrs["shap_expected"] = shap_expected
f.attrs["classes"] = list(classes)
f.attrs["columns_in_order"] = final_list
f.close()

prettycols = pd.read_csv(config.TABLES_DIR / "prettify.csv")
di = dict(zip(prettycols.ugly, prettycols.pretty_full))
di2 = dict(zip(prettycols.ugly, prettycols.pretty_abbr))
feature_names = X.columns.map(di)
feature_names_short = X.columns.map(di2)

save_feature_names_short = feature_names_short.to_list()
save_feature_names = feature_names.to_list()
f = h5py.File(save_shap_file, 'a', libver='latest')

f.attrs["feature_names_short"] = save_feature_names_short
f.attrs["feature_names"] = save_feature_names
f.close()



print("finis")
