import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from cbh import config

pd.options.mode.chained_assignment = None  # default='warn'
print("Loading", os.path.basename(__file__))

tqdm.pandas()

### read in data ###
mpn = pd.read_excel(config.RAW_AHED_MPN_FILE)
mpn["dataset_id"] = "mpn"

col_list = sorted(list(mpn))
txt_file = "zzz_mpn_raw.txt"

#  mpn_list = list(mpn)
#  list_df = pd.DataFrame(mpn_list)
#  print(list_df.head())
#  list_df.to_csv("/home/beau/dl/mpn.csv")
#  exit()


with open(config.DOCS_DIR / txt_file, "w") as f:
    f.write("MPN\n\n")
    for item in col_list:
        f.write("%s\n" % item)

mpn.rename(
    index=str,
    columns={"Female=1; male=2": "gender", "Entity": "diagnosis", "Age": "age", },
    inplace=True,
)

mpn.replace({"gender": {2: "male", 1: "female"}}, inplace=True)

### id ###

# keep only the MRN numbers
mpn["MRN"].replace(regex=True, inplace=True, to_replace=r"\D", value=r"")
# get rid of the weird non-mll-id stuff
mpn["mll_id"] = np.where(
    mpn["MLL_ID"].str.contains("MLL_", case=False, na=False), mpn["MLL_ID"], np.nan
)
# combine the columns
mpn["mll_id"].fillna(mpn["MRN"], inplace=True)
mpn["id"] = mpn["mll_id"]
mpn.drop(["mll_id", "MLL_ID", "MRN"], axis=1, inplace=True)

# convert age to numeric
mpn.age = pd.to_numeric(mpn.age, errors="coerce")

# clean up the column names and data en masse
dataframes = [mpn]
for df in dataframes:
    df.columns = df.columns.str.strip().str.lower().str.replace("  ", " ")
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)

from pandas_profiling import ProfileReport
profile = mpn.profile_report(title='Pandas Profiling Report')
profile.to_file(output_file=config.REPORTS_DIR/ "profiles/mpn.html")

# print(mpn.head())
mpn.drop(list(mpn.filter(regex="vaf")), axis=1, inplace=True)
mpn.drop(list(mpn.filter(regex="codone")), axis=1, inplace=True)
# mpn.drop(list(mpn.filter(regex="values")),axis=1, inplace=True)

keep_cols = [
    "age",
    "apc result",
    "asxl1 result",
    "asxl2 result",
    "atm result",
    "atrx result",
    "baso pb %",
    "baso pb abs",
    "bcor result",
    "bcorl1 result",
    "blasts % (bm)",
    # "blasts % (pb)",
    "braf result",
    "brcc3 result",
    "calr result",
    "cbl result",
    "cdh23 result",
    "cdkn2a result",
    "cebpa result",
    "crebbp result",
    "csf3r result",
    "csnk1a1 result",
    "ctcf result",
    "cux1 result",
    "dataset_id",
    "ddx41 result",
    "ddx54 result",
    "dhx29 result",
    "diagnosis",
    "dnmt1 result",
    "dnmt3a result",
    "eos pb %",
    "eos pb abs",
    "ep300 result",
    "etnk1 result",
    "etv6 result",
    "ezh2 result",
    "fancl result",
    "fbxw7 result",
    "flt3 result",
    "gata1 result",
    "gata2 result",
    "gender",
    "gnas result",
    "gnb1 result",
    "hb",
    "id",
    "idh1 result",
    "idh2 result",
    "jak2 result",
    "kdm5a result",
    "kdm6a result",
    "kit result",
    "kmt2d result",
    "kras result",
    "lymph pb %",
    "lymph pb abs",
    "mono pb %",
    "mono pb abs",
    "mpl result",
    "myc result",
    "neutros pb %",
    "neutros pb abs",
    "nf1 result",
    "notch1 result",
    "npm1 result",
    "nras result",
    "nsd1 result",
    "phf6 result",
    "piga result",
    "plt",
    "ppm1d result",
    "prpf8 result",
    "ptpn11 result",
    "rad21 result",
    "rb1 result",
    "runx1 result",
    "setbp1 result",
    "sf1 result",
    "sf3a1 result",
    "sf3b1 result",
    "sh2b3 result",
    "smc1a result",
    "smc3 result",
    "srsf2 result",
    "stag2 result",
    "suz12 result",
    "tet2 result",
    "tp53 result",
    "u2af1 result",
    "u2af2 result",
    "wbc",
    "wt1 result",
    "zbtb7a result",
    "zrsr2 result",
]


mpn = mpn[keep_cols]
mpn.columns = [col.replace(" result", "") for col in mpn.columns]

genes = [
    "apc",
    "asxl1",
    "asxl2",
    "atm",
    "atrx",
    "bcor",
    "bcorl1",
    "braf",
    "brcc3",
    "calr",
    "cbl",
    "cdh23",
    "cdkn2a",
    "cebpa",
    "crebbp",
    "csf3r",
    "csnk1a1",
    "ctcf",
    "cux1",
    "ddx41",
    "ddx54",
    "dhx29",
    "dnmt1",
    "dnmt3a",
    "ep300",
    "etnk1",
    "etv6",
    "ezh2",
    "fancl",
    "fbxw7",
    "flt3",
    "gata1",
    "gata2",
    "gnas",
    "gnb1",
    "idh1",
    "idh2",
    "jak2",
    "kdm5a",
    "kdm6a",
    "kit",
    "kmt2d",
    "kras",
    "mpl",
    "myc",
    "nf1",
    "notch1",
    "npm1",
    "nras",
    "nsd1",
    "phf6",
    "piga",
    "ppm1d",
    "prpf8",
    "ptpn11",
    "rad21",
    "rb1",
    "runx1",
    "setbp1",
    "sf1",
    "sf3a1",
    "sf3b1",
    "sh2b3",
    "smc1a",
    "smc3",
    "srsf2",
    "stag2",
    "suz12",
    "tet2",
    "tp53",
    "u2af1",
    "u2af2",
    "wt1",
    "zbtb7a",
    "zrsr2",
]

mpn.rename(
    index=str,
    columns={
        "baso pb %": "bas %",
        "baso pb abs": "abs bas",
        "blasts % (bm)": "percent bm blasts",
        # "blasts % (pb)": "percent pb blasts",
        "eos pb %": "eos %",
        "eos pb abs": "abs eos",
        "hb": "hgb",
        "lymph pb %": "lym %",
        "lymph pb abs": "abs lym",
        "mono pb %": "mono %",
        "mono pb abs": "abs mono",
        "neutros pb %": "neut %",
        "neutros pb abs": "abs neut",
        "plt": "plt",
        "wbc": "wbc",
    },
    inplace=True,
)

labs = [
    "bas %",
    "abs bas",
    "percent bm blasts",
    # "percent pb blasts",
    "eos %",
    "abs eos",
    "hgb",
    "lym %",
    "abs lym",
    "mono %",
    "abs mono",
    "neut %",
    "abs neut",
    "plt",
    "wbc",
]

mpn = mpn.apply(
    lambda x: x.str.lower()
    .str.strip()
    .str.replace("\t", "")
    .str.replace("  ", " ")
    .str.replace(" ", "_")
    .str.replace("__", "_")
    .str.replace("positive_2", "positive")
    .str.replace("positive2", "positive")
    .str.replace("positive_3", "positive")
    .str.replace("postiive", "positive")
    .str.replace("postivie", "positive")
    .str.replace("postitive", "positive")
    .str.replace("negativ", "negative")
    .str.replace("negativee", "negative")
    .str.replace("kmt2a", "positive")
    if (x.dtype == "object")
    else x
)

# print(list(mpn))
# for col in genes:
#     print(mpn[col].value_counts())

# for col in ["age"]:
#     mpn[col] = pd.to_numeric(mpn[col])

for col in ["id", "gender"]:
    mpn[col] = mpn[col].astype("category")

# for col in genes:
    # mpn[col] = mpn[col].astype("category")

# for col in ["date last fu"]:
#     mpn[col] = pd.to_datetime(mpn[col])
# for col in ["date of diagnosis"]:
#     mpn.drop([col], axis=1, inplace=True)

mpn["abs lym"] = mpn["abs lym"] / 1000
mpn["abs mono"] = mpn["abs mono"] / 1000
mpn["abs eos"] = mpn["abs eos"] / 1000
mpn["abs bas"] = mpn["abs bas"] / 1000
mpn["abs neut"] = mpn["abs neut"] / 1000
mpn["wbc"] = mpn["wbc"] / 1000
mpn["plt"] = mpn["plt"] / 1000

final_genes = config.FINAL_GENES
all_genes = config.GENE_COLS
for col in all_genes:
    if col not in mpn:
        mpn[col] = "missing"
    mpn[col] = mpn[col].replace(1, "positive")
    mpn[col] = mpn[col].replace(0, "negative")
    mpn[col] = mpn[col].replace(np.nan, "negative")
# print(mpn[final_genes].head(20))

col_list = sorted(list(mpn))
txt_file = "xx_mpn.txt"
# print(col_list)
with open(config.DOCS_DIR / txt_file, "w") as f:
    f.write("MPN\n\n")
    for item in col_list:
        f.write("%s\n" % item)


mpn.to_hdf(config.RAW_DATA_FILE_H5, key="mpn", mode="a", format="table")
