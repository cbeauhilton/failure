import contextlib
import os
import warnings
from functools import partial, reduce

import numpy as np
import pandas as pd

from cbh import config

warnings.simplefilter(action="ignore", category=UserWarning)
print("Loading", os.path.basename(__file__))


# f = h5py.File(config.RAW_DATA_FILE_H5, "r")
# keylist = list(f.keys())
# print(keylist)

ccf = pd.read_hdf(config.RAW_DATA_FILE_H5, key="ccf")
cmml = pd.read_hdf(config.RAW_DATA_FILE_H5, key="cmml")
mds = pd.read_hdf(config.RAW_DATA_FILE_H5, key="mds")
mpn = pd.read_hdf(config.RAW_DATA_FILE_H5, key="mpn")
icus = pd.read_hdf(config.RAW_DATA_FILE_H5, key="icus")


# MDS, ICUS, CCUS, CMML, MDS/MPN, PV, ET, PMF
# len_cohort = 0
# for df in [ccf, cmml, mds, mpn, icus]:
#     print(len(df))
#     len_cohort += len(df)
# print(len_cohort)

# for df in [ccf, cmml, mds, mpn]:
#     # print()
#     print(df.select_dtypes(include=['object']))

filename = "zyx_all.txt"
with contextlib.suppress(FileNotFoundError):
    os.remove(config.DOCS_DIR / filename)
with open(config.DOCS_DIR / filename, "a") as f:
    f.write("CCF\n\n")
    for item in sorted(list(ccf)):
        f.write(f"{item}\n")
    f.write("\n\nCMML\n\n")
    for item in sorted(list(cmml)):
        f.write(f"{item}\n")
    f.write("\n\nMDS\n\n")
    for item in sorted(list(mds)):
        f.write(f"{item}\n")
    f.write("\n\nMPN\n\n")
    for item in sorted(list(mpn)):
        f.write(f"{item}\n")
    f.write("\n\nICUS_CCUS\n\n")
    for item in sorted(list(icus)):
        f.write(f"{item}\n")


# def diff(list1, list2):
#     return list(set(list1).symmetric_difference(set(list2)))

# print(diff(sorted(list(ccf)), sorted(list(cmml))))

df0 = pd.DataFrame({"CCF": sorted(list(ccf))})
df1 = pd.DataFrame({"CMML": sorted(list(cmml))})
df2 = pd.DataFrame({"MDS": sorted(list(mds))})
df3 = pd.DataFrame({"MPN": sorted(list(mpn))})
df4 = pd.DataFrame({"ICUS_CCUS": sorted(list(icus))})
# df = pd.concat([df0, df1, df2, df3], axis=1)
df = pd.concat([df2, df4, df0, df1, df3], axis=1)
# print(df)

df.to_csv(config.DOCS_DIR / "zz_combo.csv")

final_genes = config.FINAL_GENES

# select only the columns that match between all sets
all_cols = sorted(list(set(ccf) & set(cmml) & set(mds) & set(mpn) & set(icus)))
missing_genes = set(final_genes) - set(all_cols)
if len(missing_genes) > 0:
    print("Missing genes:", missing_genes)
# print(len(all_cols))

dfs = [ccf[all_cols], cmml[all_cols], mds[all_cols], mpn[all_cols], icus[all_cols]]
outer_merge = partial(pd.merge, how="outer")
data = reduce(outer_merge, dfs)

# print(data.head(10))
# print(len(data))


data = data.apply(
    lambda x: x.str.lower()
    .str.strip()
    .str.replace("\t", "")
    .str.replace("  ", " ")
    .str.replace(" ", "_")
    .str.replace("__", "_")
    .str.replace("/", "_")
    if (x.dtype == "object")
    else x
)

# print(data.diagnosis.value_counts(dropna=False))


# MDS

data.replace(
    {
        "diagnosis": {
            "raeb-1": "mds-eb",
            "raeb-2": "mds-eb",
            "rars": "mds-rs",
            "rcmd-rs": "mds-rs",
            "rcud": "mds-sld",
            "ra": "mds-sld",
            "rcmd": "mds-mld",
        }
    },
    inplace=True,
)

data.replace(
    {
        "diagnosis": {
            "mds-eb": "mds",
            "mds-rs": "mds",
            "mds-mld": "mds",
            "mds-sld": "mds",
            "mds-u": "mds",
            "del(5q)": "mds",
            "tmds": "mds",
        }
    },
    inplace=True,
)

# MPNs
data.replace(
    {
        "diagnosis": {
            # 'mpn-u': 'mpn-u',
            # "mpn": "mpn-u",
            # 'pmf' : 'pmf',
            "mf": "pmf",
            "mf_w__myeliod_metaplasia": "pmf",
            "cimf": "pmf",
            "etmf": "et-mf",
            # "pv-mf": "?",
            # "et-mf" : "?",
            #  'mds_mf' :"?",
            # 'mf,_mds_mpn'  : "?",
            #  'et' : 'et',
            "mpn_et": "et",
            # 'et-mds' : "?",
            #  'pv' : 'pv',
            # 'pv-mpn,_cll' : "?",
            # 'mf_cml' : '?',
            # 'cml' : '?',
        }
    },
    inplace=True,
)

# MDS/MPNs
data.replace(
    {
        "diagnosis": {
            "cmml-1": "cmml",
            "cmml-2": "cmml",
            "rars-t": "mds-mpn-rs-t",
            "mpn-cmml": "cmml",
            # 'cmml_cll' : 'cmml?'
        }
    },
    inplace=True,
)
data.replace(
    {
        "diagnosis": {
            # 'mds_mpn' : 'mds_mpn',
            "mds_mpn-u": "mds_mpn",
            "mds-mpn-rs-t": "mds_mpn",
            "mpn_mds-u": "mds_mpn",
            "mpn_or_mds": "mds_mpn",
            "mds_cmml": "mds_mpn",
            "mds_mpv": "mds_mpn",
        }
    },
    inplace=True,
)

# AMLs
# data.replace(
#     {
#         "diagnosis": {
#             'aml' : 'aml',
#             'cml_aml' : 'aml',
#             'pv_aml' : 'aml',
#             'mpn_aml' : 'aml',
#             'saml' : 'aml',
#             'mf_aml' : 'aml',
#         }
#     },
#     inplace=True,
# )


# Other
# 'pnh'
# 'mpd'


# print(data.diagnosis.value_counts(dropna=False))

# get rid of rare diagnoses
col = "diagnosis"
n = 70
data = data[data.groupby(col)[col].transform("count").ge(n)]


print(data.diagnosis.value_counts(dropna=False))

# MDS, ICUS, CCUS, CMML, MDS/MPN, PV, ET, PMF

# print(data.diagnosis.unique())

# print(data.diagnosis.value_counts(dropna=False))
# print(len(data.diagnosis.notnull()))
# print(data.count())


data.columns = [col.replace("%", "percent") for col in data.columns]
data.columns = [col.replace(" ", "_") for col in data.columns]

final_genes = config.FINAL_GENES
final_genes = config.GENE_COLS
for col in final_genes:
    if col not in data:
        # print(col)
        data[col] = "missing"
    data[col] = data[col].replace(1, "positive")
    data[col] = data[col].replace("potitive", "positive")
    data[col] = data[col].replace(0, "negative")
    data[col] = data[col].replace(np.nan, "missing")

cats = [
    # "cebpa",
    "gender",
    # "kdm6a",
    # "nf1",
    # "phf6",
    # "prpf8",
    # "ptpn11",
    # "rad21",
    # "smc3",
    # "stag2",
    # "u2af2",
    # "wt1",
]

# genes = [
#     "asxl1",
#     "bcor",
#     "cbl",
#     "dnmt3a",
#     "etv6",
#     "ezh2",
#     "flt3",
#     "gata2",
#     "idh1",
#     "idh2",
#     "jak2",
#     "kit",
#     "kras",
#     "npm1",
#     "nras",
#     "runx1",
#     "sf3b1",
#     "srsf2",
#     "tet2",
#     "tp53",
#     "u2af1",
#     "zrsr2",
# ]

cat_list = []
for col in cats:
    cat_list.append([col1 for col1 in data.columns if col.lower() in col1])
cat_list.append(final_genes)
cat_list = [item for sublist in cat_list for item in sublist]
# print(cat_list)


data["id"] = data["id"].astype("category")
# print(data["id"])

# print(data[["institution", "id"]])
data[cat_list] = data[cat_list].astype("category")
one_hot = pd.get_dummies(data[cat_list])
# Drop columns now encoded
data = data.drop(cat_list, axis=1)
# Join the encoded df
data = data.join(one_hot)

negative_cols = [col for col in data.columns if "_negative" in col]
missing_cols = [col for col in data.columns if "_missing" in col]
# print(len(missing_cols))
# print(missing_cols)
# print(negative_cols)
data = data.drop(negative_cols, axis=1)
data = data.drop(missing_cols, axis=1)

# drop any columns that are completely blank
with_na = len(list(data))
data = data.dropna(axis=1, how="all")
without_na = len(list(data))
any_dropped = with_na - without_na
if any_dropped > 0:
    print(f"Dropped {any_dropped} empty columns.")

# Drop rows missing key data
# print(len(data))
data = data[pd.notnull(data["age"])]
# print(len(data))
data = data[pd.notnull(data["wbc"])]
# print(len(data))

# catch a few more unit problems with labs
# print(data.abs_neut.describe())
# data["abs_neut"] = data["abs_neut"].apply(lambda x: x / 1000 if x > 200 else x)
# data["abs_lym"] = data["abs_lym"].apply(lambda x: x / 1000 if x > 200 else x)
# print(data.abs_neut.describe())

# print(data.wbc.describe())
# data["wbc"] = data["wbc"].apply(lambda x: x / 1000 if x > 200 else x)
# print(data.wbc.describe())

# attempt to get a few more labs if calculatable
abs_cols = list([col for col in data.columns if "abs_" in col])
per_cols = list([col for col in data.columns if "_percent" in col])

data["bas_abs_calc"] = data["wbc"] * data["bas_percent"]
data["eos_abs_calc"] = data["wbc"] * data["eos_percent"]
data["lym_abs_calc"] = data["wbc"] * data["lym_percent"]
data["mono_abs_calc"] = data["wbc"] * data["mono_percent"]
data["neut_abs_calc"] = data["wbc"] * data["neut_percent"]

abs_calc_cols = list([col for col in data.columns if "_abs_calc" in col])

# print(abs_cols)
# print(abs_calc_cols)
# print(data[data.abs_bas_calc.notnull()])

for raw, calc in zip(abs_cols, abs_calc_cols):
    data[raw] = data[raw].fillna(data[calc])

for raw in abs_cols:
    data[raw].replace(0, np.nan)

data["bas_percent_calc"] = data["abs_bas"] / data["wbc"]
data["eos_percent_calc"] = data["abs_eos"] / data["wbc"]
data["lym_percent_calc"] = data["abs_lym"] / data["wbc"]
data["mono_percent_calc"] = data["abs_mono"] / data["wbc"]
data["neut_percent_calc"] = data["abs_neut"] / data["wbc"]

per_calc_cols = list([col for col in data.columns if "_percent_calc" in col])

# print(per_cols)
# print(per_calc_cols)
for raw, calc in zip(per_cols, per_calc_cols):
    data[raw] = data[raw].fillna(data[calc])

for raw in per_cols:
    data[raw].replace(0, np.nan)

# Drop columns with high percentage of missing data
missing_percent = 0.30
high_nan = list(data.loc[:, data.isnull().mean() > missing_percent])

# for col in high_nan:
#     print(col, data[col].isnull().mean())

# print(len(list(data)))
data = data.loc[:, data.isnull().mean() < missing_percent]
# print(len(list(data)))

if len(high_nan) > 0:
    print(f"Dropped the cols {high_nan}")

# print(list(data))
# data = data[pd.notnull(data['abs_neut'])]
data = data.drop(per_cols, axis=1)
calc_cols = list([col for col in data.columns if "_calc" in col])
data = data.drop(calc_cols, axis=1)
data = data.drop("gender_male", axis=1)
add_gene_cols = [col for col in list(data) if "_positive" in col]
data["mutation_num"] = data[add_gene_cols].sum(axis=1)
data.loc[
    (data["diagnosis"] == "not_yet_mds") & (data["mutation_num"] > 0), "diagnosis"
] = "ccus"
data.loc[
    (data["diagnosis"] == "not_yet_mds") & (data["mutation_num"] == 0), "diagnosis"
] = "icus"
# print(data.diagnosis.unique())

data = data.copy()

# for MDS vs ICUS and CCUS
# data = data[data['diagnosis'].isin(["mds", "ccus", "icus"])]

data = data.reset_index(drop=True)

# Grab indices to make new identifiers
data["pt_num"] = data.index
data["pt_num"] = data["pt_num"].astype("category")
# print(data.diagnosis.value_counts(dropna=False))

# print(len(data))
# print(list(data))
data.to_hdf(config.RAW_DATA_FILE_H5, key="data", mode="a", format="table")
data.to_csv(config.INTERIM_DATA_DIR / "combo.csv")

y = data["diagnosis"].copy()
X = data.drop(["diagnosis", "id", "dataset_id"], axis=1).copy()

y.to_hdf(config.RAW_DATA_FILE_H5, key="y", mode="a", format="table")
X.to_hdf(config.RAW_DATA_FILE_H5, key="X", mode="a", format="table")
