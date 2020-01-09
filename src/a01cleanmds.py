import os

import numpy as np
import pandas as pd

from cbh import config

pd.options.mode.chained_assignment = None  # default='warn'
print("Loading", os.path.basename(__file__))
### read in data ###
mds = pd.read_excel(config.RAW_MLL_MDS_FILE)
mds["dataset_id"] = "mds"

col_list = sorted(list(mds))
txt_file = "zzz_mds_raw.txt"

#  mds_list = list(mds)
#  list_df = pd.DataFrame(mds_list)
#  print(list_df.head())
#  list_df.to_csv("/home/beau/dl/mds.csv")
#  exit()

with open(config.DOCS_DIR / txt_file, "w") as f:
    f.write("MDS\n\n")
    for item in col_list:
        f.write("%s\n" % item)


# clean up the column names and data en masse
dataframes = [mds]
for df in dataframes:
    df.columns = df.columns.str.strip().str.lower().str.replace("  ", " ")
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)

mds.rename(
    index=str,
    columns={"mll_id": "id", "who": "diagnosis", "anc": "abs neut", "hb": "hgb", "plts": "plt"},
    inplace=True,
)

# #XX and XY to male and female
mds.loc[mds["cytogenetic"].str.contains("XY", case=False), "gender"] = "male"
mds.loc[mds["cytogenetic"].str.contains("X,-Y", case=False), "gender"] = "male"
mds.loc[mds["cytogenetic"].str.contains("XX", case=False), "gender"] = "female"
mds.loc[mds["cytogenetic"].str.contains("X,-X", case=False), "gender"] = "female"

# print(mds.shape)


# #drop redundancies and other unwanted columns
# print(mds.shape)
mds = mds[mds.columns.drop(list(mds.filter(regex="vaf")))]
mds = mds[
    mds.columns.drop(
        list(
            mds.filter(
                items=[
                    "race",
                    #'wbc',
                    #   'mono %', 'lym %','neut %', 'eos %', 'bas %',
                    "cytogenetic",
                ]
            )
        )
    )
]
# print(mds.shape)

genes = [
    "asxl1",
    "bcor",
    "bcorl1",
    "cbl",
    "cebpa",
    "dnmt3a",
    "etv6",
    "ezh2",
    "flt3",
    "gata2",
    "gnb1",
    "idh1",
    "idh2",
    "jak2",
    "kdm6a",
    "kit",
    "kras",
    "luc7l2",
    "med12",
    "mpl",
    "nf1",
    "notch1",
    "npm1",
    "nras",
    "phf6",
    "prpf40b",
    "prpf8",
    "ptpn11",
    "rad21",
    "runx1",
    "sf3b1",
    "smc3",
    "srsf2",
    "stag2",
    "tet2",
    "tp53",
    "u2af1",
    "u2af2",
    "wt1",
    "zrsr2",
]

# fix scale to match CCF dataset
# mds["abs lym"] = mds["abs lym"] / 1000
# mds["abs mono"] = mds["abs mono"] / 1000
# mds["abs eos"] = mds["abs eos"] / 1000
# mds["abs bas"] = mds["abs bas"] / 1000

mds["abs lym"] = mds["abs lym"] / 1000
mds["abs mono"] = mds["abs mono"] / 1000
mds["abs eos"] = mds["abs eos"] / 1000
mds["abs bas"] = mds["abs bas"] / 1000
mds["abs neut"] = mds["abs neut"] / 1000
mds["wbc"] = mds["wbc"] / 1000
mds["plt"] = mds["plt"] / 1000

final_genes = config.FINAL_GENES
all_genes = config.GENE_COLS
for col in all_genes:
    if col not in mds:
        mds[col] = "missing"
    mds[col] = mds[col].replace(1, "positive")
    mds[col] = mds[col].replace(0, "negative")
    mds[col] = mds[col].replace(np.nan, "missing")

# print(mds[final_genes].head(20))



col_list = sorted(list(mds))
txt_file = "xx_mds.txt"
# print(col_list)
with open(config.DOCS_DIR / txt_file, "w") as f:
    f.write("MDS\n\n")
    for item in col_list:
        f.write("%s\n" % item)


mds.to_hdf(config.RAW_DATA_FILE_H5, key="mds", mode="a", format="table")
