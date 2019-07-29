import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from cbh import config

pd.options.mode.chained_assignment = None  # default='warn'
print("Loading", os.path.basename(__file__))

tqdm.pandas()

### read in data ###
icus = pd.read_excel(config.RAW_MLL_ICUS_CCUS_FILE)
icus["dataset_id"] = "icus_ccus"


col_list = sorted(list(icus))
txt_file = "zzz_icus_raw.txt"

with open(config.DOCS_DIR / txt_file, "w") as f:
    f.write("ICUS_CCUS\n\n")
    for item in col_list:
        f.write("%s\n" % item)

icus.rename(
    index=str,
    columns={
        "Gender[1=XX, 2=XY]": "gender",
        "MDS WHO 2008": "diagnosis",
        "Age [yrs]": "age",
        "WBC [/µl]": "wbc",
        "Hemoglobin [G7DL]": "hgb",
        "MLL_ID": "id",
    },
    inplace=True,
)



result_cols = list([col for col in icus.columns if " result" in col])

for col in result_cols:
    icus[col][icus[col].str.contains("positive") == True] = "positive"
    icus[col][icus[col].str.contains("fraglich") == True] = "vus"  # Germans!
    icus[col][icus[col].str.contains("variant") == True] = "vus"
    icus[col][icus[col].str.contains("n.a") == True] = "missing"
    icus[col][icus[col].str.contains("-") == True] = "missing"
    icus[col] = icus[col].replace(np.nan, "missing")



ccus_genes = [
    "ASXL1",  ###
    "BCOR",
    "CALR",  # <2% MDS, 30 PMF, 30 ET, NR for CHIP
    "CBL",
    "DNMT3A",  ###
    "ETV6",
    "EZH2",
    "FLT3",
    "GATA2",  # <1% freq in MDS, very rare in other heme malig
    "GNAS",  # <2%, 4% CHIP, very rare in other heme malig
    "GNB1",  # <2%, 4% CHIP, very rare in other heme malig
    "IDH1",
    "IDH2",
    "JAK2",
    "KIT",
    "KRAS",
    "MPL",
    "NPM1",
    "NRAS",
    "PHF6",
    "RAD21",  # <2% MDS, 4% AML, <1% CHIP
    "RUNX1",
    "SETBP1",
    "SF3B1",
    "SMC3",  # <2 MDS, <4 AML, <1 CHIP
    "SRSF2",
    "STAG2",  # 2-5% in MDS, rare in AML (4%), <1% in CHIP
    "TET2",  ###
    "TP53",
    "U2AF1",
    "ZRSR2",
]




icus.replace({"gender": {2: "male", 1: "female"}}, inplace=True)

# # convert age to numeric
icus.age = pd.to_numeric(icus.age, errors="coerce")

# # clean up the column names and data en masse
# dataframes = [icus]
# for df in dataframes:
icus.columns = icus.columns.str.strip().str.lower().str.replace("  ", " ")
icus = icus.applymap(lambda s: s.lower() if type(s) == str else s)

gene_cols = []
for col in ccus_genes:
    col_list = list([col1 for col1 in icus.columns if f"{col.lower()} result" in col1])
    gene_cols.append(col_list)

# print(gene_cols)

flat_gene_list = [item for sublist in gene_cols for item in sublist]
# print(flat_gene_list)
# print(len(ccus_genes))

# Bunch of silly stuff to deal with unicode in the column names - 
# could probably be done in a one-liner, but I'm too dumb for that
encode_dict = {}
for i, item in enumerate(list(icus)):
    # print(item)
    item1 = item.encode('ascii', 'ignore').decode("utf-8")
    encode_dict[item] = item, item1, i
# print(encode_dict)
uni_list = []
decode_list = []
old_name_list = []
for k, (val1, val2, val3) in encode_dict.items():
    if val1 != val2:
        uni_list.append(val2)
        decode_list.append(val3)
        old_name_list.append(val1)
    # print(val1)
    # print(val2)
# print("Uni:", uni_list)
# print("Decode:", decode_list)
check_uni = icus.iloc[:,decode_list]
# print(check_uni)
for new_col, old_col, old_name in zip(uni_list, decode_list, old_name_list):
    icus[new_col] = icus[icus.columns[old_col]]
    if icus[new_col].equals(check_uni[old_name]):
        pass
    else:
        print("Columns not equal!")
icus.drop(icus.columns[decode_list], axis=1, inplace=True)


# print(list(icus))

keep_cols = [
    "id",
    "age",
    "gender",
    "diagnosis",
    "wbc",
    "anc",
    "hgb",
    "plt [/l]",
    "pb neut %",
    "pb neut abs",
    "pb lymph %",
    "pb lymph abs",
    "pb mono %",
    "pb mono abs",
    "pb eos %",
    "pb eos abs",
    "pb baso %",
    "pb baso abs",
    "dataset_id",
]

keep_cols.extend(flat_gene_list)

# print(keep_cols)

icus = icus[keep_cols]
icus.columns = [col.replace(" result", "") for col in icus.columns]

icus.rename(
    index=str,
    columns={
        "pb baso %": "bas %",
        "pb baso abs": "abs bas",
        "blasts % (bm)": "percent bm blasts",
        # "blasts % (pb)": "percent pb blasts",
        "pb eos %": "eos %",
        "pb eos abs": "abs eos",
        "hb": "hgb",
        "pb lymph %": "lym %",
        "pb lymph abs": "abs lym",
        "pb mono %": "mono %",
        "pb mono abs": "abs mono",
        "pb neut %": "neut %",
        "pb neut abs": "abs neut",
        "plt [/l]": "plt",
        "wbc": "wbc",
    },
    inplace=True,
)

icus = icus.apply(
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

# # print(list(icus))

genes = []
for col in ccus_genes:
    genes.append([col1 for col1 in icus.columns if col.lower() in col1])

genes = [item for sublist in genes for item in sublist]

# for col in genes:
#     print(icus[col].value_counts())

for col in ["id", "gender"]:
    icus[col] = icus[col].astype("category")

# for col in genes:
    # icus[col] = icus[col].astype("category")


icus["abs lym"] = icus["abs lym"] / 1000
icus["abs mono"] = icus["abs mono"] / 1000
icus["abs eos"] = icus["abs eos"] / 1000
icus["abs bas"] = icus["abs bas"] / 1000
icus["abs neut"] = icus["abs neut"] / 1000
icus["wbc"] = icus["wbc"] / 1000
icus["plt"] = icus["plt"] / 1000

labs_list = [
'hgb', 'abs bas', 'abs eos', 'abs lym', 'abs mono', 'abs neut',
]

missing_percent = 0.10

high_nan = list(icus.loc[:, icus.isnull().mean() > missing_percent])
# print(high_nan)

# for col in labs_list:
#     # print(col)
#     try:
#         icus[col].fillna((icus[col].median()), inplace=True)
#     except:
#         print(f"Could not impute for column {col} ")

all_genes = config.GENE_COLS
for col in all_genes:
    if col not in icus:
        icus[col] = "missing"

col_list = sorted(list(icus))
# print(col_list)
txt_file = "xx_icus_ccus.txt"
# print(col_list)
with open(config.DOCS_DIR / txt_file, "w") as f:
    f.write("ICUS_CCUS\n\n")
    for item in col_list:
        f.write("%s\n" % item)


icus.to_hdf(config.RAW_DATA_FILE_H5, key="icus", mode="a", format="table")
