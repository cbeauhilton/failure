import pandas as pd
from cbh import config
import numpy as np

data = pd.read_hdf(config.RAW_DATA_FILE_H5, key="data")

# Define labels and features
y = data["diagnosis"].copy()
X = data.drop(["diagnosis", "id"], axis=1).copy()
classes = np.unique(y) # retrieve all class names
CLASSES = [x.upper() for x in classes] # make uppercase version

prettycols = pd.DataFrame()

prettycols["ugly"] = list(data)
prettycols["pretty_full"] = prettycols["ugly"]
prettycols["pretty_abbr"] = prettycols["ugly"]



# prettycols_file = config.PRETTIFYING_COLUMNS_CSV
# prettycols = pd.read_csv(prettycols_file)
# di = dict(zip(prettycols.feature_ugly, prettycols.feature_pretty))
# pretty_imp_cols = imp_cols[0].map(di).fillna(imp_cols[0])

prettycols["pretty_full"].replace(
    {
        "mono_percent": "Monocyte percentage",
        "abs_mono": "Absolute monocyte count",
        "age": "Age",
        "abs_neut": "Absolute neutrophil count",
        "hgb": "Hemoglobin",
        "abs_lym": "Absolute lymphocyte count",
        "wbc": "White blood cell count",
        "neut_percent": "Neutrophil percentage",
        "lym_percent": "Lymphocyte percentage",
        "eos_percent": "Eosinophil percentage",
        "abs_eos": "Absolute eosinophil count",
        "bas_percent": "Basophil percentage",
        "abs_bas": "Absolute basophil count",
        "gender_male": "Male",
    },
    inplace=True,
)

prettycols["pretty_abbr"].replace(
    {
        "mono_percent": "Monocyte percentage",
        "abs_mono": "AMC",
        "age": "Age",
        "abs_neut": "ANC",
        "hgb": "Hgb",
        "abs_lym": "ALC",
        "wbc": "WBC",
        "neut_percent": "Neutrophil percentage",
        "lym_percent": "Lymphocyte percentage",
        "eos_percent": "Eosinophil percentage",
        "abs_eos": "AEC",
        "bas_percent": "Basophil percentage",
        "abs_bas": "ABC",
        "gender_male": "Male",
    },
    inplace=True,
)

print(prettycols.head())


prettycols.to_csv(config.TABLES_DIR / "prettify.csv")


imp_cols = pd.read_csv(config.TABLES_DIR / "shap_df.csv")
prettycols = pd.read_csv(config.TABLES_DIR / "prettify.csv")
di = dict(zip(prettycols.ugly, prettycols.pretty_full))
di2 = dict(zip(prettycols.ugly, prettycols.pretty_abbr))
pretty_imp_cols = pd.DataFrame()
for classname in CLASSES:
    pretty_imp_cols[f'{classname}'] = imp_cols[f'{classname}'].map(di).fillna(imp_cols[f'{classname}'])
    pretty_imp_cols[f'{classname}_full'] = imp_cols[f'{classname}'].map(di).fillna(imp_cols[f'{classname}'])
    pretty_imp_cols[f'{classname}_abbr'] = imp_cols[f'{classname}'].map(di2).fillna(imp_cols[f'{classname}'])
print(pretty_imp_cols.head())

pretty_imp_cols.to_csv(config.TABLES_DIR / "shap_df_pretty.csv")