import os

import numpy as np
import pandas as pd

from cbh import config

pd.options.mode.chained_assignment = None  # default='warn'
print("Loading", os.path.basename(__file__))
### read in data ###
cmml = pd.read_excel(config.RAW_MLL_CMML_FILE)

col_list = sorted(list(cmml))
txt_file = "zzz_cmml_raw.txt"

with open(config.DOCS_DIR/txt_file, "w") as f:
    f.write("CMML\n\n")
    for item in col_list:
        f.write("%s\n" % item)

# clean up the column names and data en masse
dataframes = [cmml]
for df in dataframes:
    df.columns = df.columns.str.strip().str.lower().str.replace("  ", " ")
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)

### column names ###

cmml.rename(
    index=str,
    columns={
        "patientid": "id",
        "who category": "diagnosis",
        "date of diagnosis/sampling": "date of diagnosis or sampling",
        "date of last contact/death": "date of last contact or death",
        "# cytopenia": "number of cytopenias",
    },
    inplace=True,
)

### gender ###
cmml.replace({"gender": {1: "male", 2: "female"}}, inplace=True)


# print(cmml.head())


# Drop stuff
# print(cmml.shape)

cmml = cmml[cmml.columns.drop(list(cmml.filter(regex="position")))]
cmml = cmml[cmml.columns.drop(list(cmml.filter(regex="result")))]
cmml = cmml[cmml.columns.drop(list(cmml.filter(regex="change")))]
cmml = cmml[cmml.columns.drop(list(cmml.filter(regex="sidroblasts")))]  # sic
cmml = cmml[cmml.columns.drop(list(cmml.filter(regex="chr")))]
cmml = cmml[cmml.columns.drop(list(cmml.filter(regex="consequence")))]
cmml = cmml[cmml.columns.drop(list(cmml.filter(regex="cellularity")))]
cmml = cmml[cmml.columns.drop(list(cmml.filter(regex="double")))]
cmml = cmml[cmml.columns.drop(list(cmml.filter(regex="poiesis")))]
cmml = cmml[cmml.columns.drop(list(cmml.filter(regex="reticulin fibrosis")))]
cmml = cmml[cmml.columns.drop(list(cmml.filter(regex="vaf")))]

# print(cmml.shape)


cmml.rename(
    {
        f"blast% bmbx": "percent bm blasts",
        f"circulating blasts (%)": "percent circulating blasts",
        f"circulating imc (%)": "percent circulating imc",
        "ferritin >1000": "ferritin over 1000",
        "age at dx": "age",
        "hgb at dx": "hgb",
        "wbc at dx": "wbc",
        "platelets at dx": "plt",
        # "monocytes at dx": "abs mono",
        "lymphocytes at dx": "abs lym",
    },
    axis="columns",
    inplace=True,
)
# print(list(cmml))


cmml["splenomegaly"].fillna(0, inplace=True)

# print(cmml.shape)

# fix scale to match CCF database
cmml["abs lym"] = cmml["abs lym"] / 1000
cmml["abs mono"] = cmml["abs mono"] / 1000
cmml["abs eos"] = cmml["abs eos"] / 1000
cmml["abs bas"] = cmml["abs bas"] / 1000
cmml["abs neut"] = cmml["abs neut"] / 1000
cmml["wbc"] = cmml["wbc"] / 1000
cmml["plt"] = cmml["plt"] / 1000

# print(sorted(list(cmml)))
for col in ["gender", "diagnosis", "id"]:
    cmml[col] = cmml[col].astype("category")


# print(cmml.head(10))

col_list = sorted(list(cmml))
txt_file = "zz_cmml.txt"

with open(config.DOCS_DIR/txt_file, "w") as f:
    f.write("CMML\n\n")
    for item in col_list:
        f.write("%s\n" % item)


cmml.to_hdf(config.RAW_DATA_FILE_H5, key="cmml", mode="a", format="table")
