import os

import numpy as np
import pandas as pd

from cbh import config

pd.options.mode.chained_assignment = None  # default='warn'
print("Loading", os.path.basename(__file__))
### read in data ###
ccf = pd.read_csv(config.RAW_CCF_FILE)

col_list = sorted(list(ccf))
txt_file = "zzz_ccf_raw.txt"

with open(config.DOCS_DIR/txt_file, "w") as f:
    f.write("CCF\n\n")
    for item in col_list:
        f.write("%s\n" % item)
            
# print(list(ccf))
# print("")
### column names ###
ccf.rename(index=str, columns={"GENDER": "gender"}, inplace=True)

### id ###
# id ccf #
ccf["id"] = "CCF_" + ccf.index.astype(str)

ccf.columns = ccf.columns.str.strip().str.lower().str.replace("  ", " ")
ccf = ccf.applymap(lambda s: s.lower() if type(s) == str else s)


ccf = ccf[ccf.columns.drop(list(ccf.filter(regex="poiesis")))]
ccf = ccf[ccf.columns.drop(list(ccf.filter(regex="vaf")))]
# print(list(ccf))
ccf.rename(
    {"who category": "diagnosis", "male=0": "gender"}, axis="columns", inplace=True
)

ccf["splenomegaly"].replace({"yes": 1, "no": 0}, inplace=True)

ccf["gender"].replace({0: "male", 1: "female"}, inplace=True)

ccf = ccf[
    ccf.columns.drop(
        list(
            ccf.filter(
                items=[
                    # "mono %",
                    # "lym %",
                    # "neut %",
                    # "eos %",
                    # "bas %",
                    "cellularity",
                    #'percent bm blasts',
                    "cytogenetic points - ipss-r",
                    "tfx dependent =1",
                    "ecog ps at dx",
                    "per blasts",
                ]
            )
        )
    )
]



for col in ["gender", "megakaryocytes", "diagnosis", "id"]:
    ccf[col] = ccf[col].astype("category")


# print(ccf.head(10))

col_list = sorted(list(ccf))
txt_file = "zz_ccf.txt"

with open(config.DOCS_DIR/txt_file, "w") as f:
    f.write("CCF\n\n")
    for item in col_list:
        f.write("%s\n" % item)


ccf.to_hdf(config.RAW_DATA_FILE_H5, key="ccf", mode="a", format="table")
