import contextlib
import os
import warnings
from functools import partial, reduce

import h5py
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
        f.write("%s\n" % item)
    f.write("\n\nCMML\n\n")
    for item in sorted(list(cmml)):
        f.write("%s\n" % item)
    f.write("\n\nMDS\n\n")
    for item in sorted(list(mds)):
        f.write("%s\n" % item)
    f.write("\n\nMPN\n\n")
    for item in sorted(list(mpn)):
        f.write("%s\n" % item)
    f.write("\n\nICUS_CCUS\n\n")
    for item in sorted(list(icus)):
        f.write("%s\n" % item)


# def diff(list1, list2):
#     return list(set(list1).symmetric_difference(set(list2)))

# print(diff(sorted(list(ccf)), sorted(list(cmml))))

df0 = pd.DataFrame({"CCF": sorted(list(ccf))})
df1 = pd.DataFrame({"CMML": sorted(list(cmml))})
df2 = pd.DataFrame({"MDS": sorted(list(mds))})
df3 = pd.DataFrame({"MPN": sorted(list(mpn))})
df4 = pd.DataFrame({"ICUS_CCUS": sorted(list(icus))})
# df = pd.concat([df0, df1, df2, df3], axis=1)
df = pd.concat([df0, df1, df2, df3, df4], axis=1)
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

print(data.diagnosis.value_counts(dropna=False))


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
            'cimf' : "pmf",
            'etmf' : 'et-mf',
            # "pv-mf": "?",
            # "et-mf" : "?",
            #  'mds_mf' :"?",
            # 'mf,_mds_mpn'  : "?",
            #  'et' : 'et', 
            'mpn_et' : 'et',
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
 

print(data.diagnosis.value_counts(dropna=False))

# get rid of rare diagnoses
col = "diagnosis"
n = 10
data = data[data.groupby(col)[col].transform("count").ge(n)]


print(data.diagnosis.value_counts(dropna=False))


# print(data.diagnosis.unique())

# print(data.diagnosis.value_counts(dropna=False))
# print(len(data.diagnosis.notnull()))
# print(data.count())


data.columns = [col.replace("%", "percent") for col in data.columns]
data.columns = [col.replace(" ", "_") for col in data.columns]

final_genes = config.FINAL_GENES

for col in final_genes:
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

genes = [
    "asxl1",
    "bcor",
    "cbl",
    "dnmt3a",
    "etv6",
    "ezh2",
    "flt3",
    "gata2",
    "idh1",
    "idh2",
    "jak2",
    "kit",
    "kras",
    "npm1",
    "nras",
    "runx1",
    "sf3b1",
    "srsf2",
    "tet2",
    "tp53",
    "u2af1",
    "zrsr2",
]

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
# print(negative_cols)
data = data.drop(negative_cols, axis=1)

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

data = data.reset_index(drop=True)
# who needs all dem decimals


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

# #just in case...
# combo.columns = combo.columns.str.strip().str.lower().str.replace('  ', ' ')
# combo = combo.applymap(lambda s:s.lower() if type(s) == str else s)

# #fix splenomegaly and gender columns
# combo.replace({
#                 'yes' : 1,
#                 'no' : 0
#             }, inplace=True)

# combo.replace({
#                 'male' : 0,
#                 'female' : 1
#             }, inplace=True)


# # drop unwanted columns
# combo = combo[combo.columns.drop(list(combo.filter(items=[


#                                                          'megakaryocytes',
#                                                          'rdw','mch','mchc','mcv', 'mpv',
#                                                          'splenomegaly'

#                                                          ])))]

# #create new feature called "total number of mutations"
# combo['total number of mutations'] =  combo[['apc',
#        'asxl1', 'bcor', 'bcorl1', 'braf', 'c7orf55', 'cbl', 'ccdc42b', 'cdh23',
#        'cdkn2a', 'cebpa', 'cftr', 'csf1r', 'cux1', 'ddx41', 'ddx54', 'dhx29',
#        'dnmt3a', 'eed', 'erbb4', 'etv6', 'ezh2', 'flt3', 'gata2',
#        'gli1', 'gli2', 'gnb1', 'gpr98', 'idh1', 'idh2', 'irf4', 'jak2',
#        'jak3', 'kdm5a', 'kdm6a', 'kit', 'kras', 'luc7l2',
#        'mecom', 'med12', 'mll', 'mpl', 'nf1',
#        'notch1', 'npm1', 'nras', 'ogt', 'phf6', 'prpf40b', 'prpf8',
#        'ptch1', 'pten', 'ptpn11', 'rad21', 'rnf25', 'runx1', 'setbp1',
#        'sf1', 'sf3b1', 'simc1', 'smc3','srsf2', 'stag2',
#        'stat3', 'suz12', 'tet2', 'tp53', 'u2af1', 'u2af2',
#         'wt1', 'zrsr2']].sum(axis=1)


# MDS_data = combo

# # print('All Subclasses:\n', MDS_data['who category'].value_counts())


# # remove pts w BM blast % >20
# MDS_data.drop(MDS_data[MDS_data['percent bm blasts'] > 20].index, inplace=True)

# # fix mislabeled rars-t patients
# MDS_data.loc[(MDS_data['plt'] > 450) & (MDS_data['who category']=='rars'),'who category'] = 'rars-t'
# # print('All Subclasses _fixed rars-t_:\n', MDS_data['who category'].value_counts())

# # condense subtypes into two categories: MDS and CMML
# # first update naming convention, if needed later
# # most important correction is RARS-T to MDS/MPN-RS-T, which will be dropped from the analysis
# MDS_data.replace({'who category': {'raeb-1': 'mds-eb', 'raeb-2': 'mds-eb'}}, inplace=True)
# MDS_data.replace({'who category': {'rars': 'mds-rs', 'rars-t': 'mds-mpn-rs-t', 'rcmd-rs': 'mds-rs'}}, inplace=True)
# MDS_data.replace({'who category': {'cmml-1': 'cmml', 'cmml-2': 'cmml'}}, inplace=True)
# MDS_data.replace({'who category': {'rcud': 'mds-sld', "ra": 'mds-sld'}}, inplace=True)
# MDS_data.replace({'who category': {'rcmd': 'mds-mld'}}, inplace=True)
# MDS_data.replace({'who category': {'mds-eb' : 'mds', 'mds-rs' : 'mds', 'mds-mld': 'mds', 'mds-sld': 'mds',
#                                   'mds-u': 'mds', 'del(5q)': 'mds', 'tmds': 'mds'}}, inplace=True)

# #drop mds/mpn-u and mds-mpn-rs-t, make CMML and MDS datasets for descriptive statistics
# MDS_data = MDS_data[MDS_data['who category'].str.contains("mds/mpn-u") == False]
# # MDS_data = MDS_data[MDS_data['who category'].str.contains("mds-mpn-rs-t") == False]
# cmml_only = MDS_data[MDS_data['who category'].str.contains("mds") == False]
# mds_only = MDS_data[MDS_data['who category'].str.contains("cmml") == False]

# print('MDS and CMML Only: \n', MDS_data['who category'].value_counts())
# # print('MDS Only: \n', mds_only['who category'].value_counts())
# # print('CMML Only: \n', cmml_only['who category'].value_counts())


# # # descriptive statistics
# # writer = pd.ExcelWriter('01 MDS descriptive.xlsx')
# # MDS_data.describe().to_excel(writer,'Sheet1')
# # writer.save()

# # MDS_data.describe()
# # cmml_only.describe()
# # mds_only.describe()


# #Get percentages of mutations

# mds_only.fillna(0, inplace=True)
# cmml_only.fillna(0, inplace=True)

# mds_cmml_features = MDS_data.drop(["who category"], axis = 1)
# mds_features = mds_only.drop(["who category"], axis = 1)
# cmml_features = cmml_only.drop(["who category"], axis = 1)

# percentages_mds_cmml = mds_cmml_features.apply(pd.Series.sum)/len(mds_cmml_features)
# percentages_mds = mds_features.apply(pd.Series.sum)/len(mds_features)
# percentages_cmml = cmml_features.apply(pd.Series.sum)/len(cmml_features)

# percentages_genes_mds_cmml = percentages_mds_cmml[[
#         'apc',
#        'asxl1', 'bcor', 'bcorl1', 'braf', 'c7orf55', 'cbl', 'ccdc42b', 'cdh23',
#        'cdkn2a', 'cebpa', 'cftr', 'csf1r', 'cux1', 'ddx41', 'ddx54', 'dhx29',
#        'dnmt3a', 'eed', 'erbb4', 'etv6', 'ezh2', 'flt3', 'gata2',
#        'gli1', 'gli2', 'gnb1', 'gpr98', 'idh1', 'idh2', 'irf4', 'jak2',
#        'jak3', 'kdm5a', 'kdm6a', 'kit', 'kras', 'luc7l2',
#        'mecom', 'med12', 'mll', 'mpl', 'nf1',
#        'notch1', 'npm1', 'nras', 'ogt', 'phf6', 'prpf40b', 'prpf8',
#        'ptch1', 'pten', 'ptpn11', 'rad21', 'rnf25', 'runx1', 'setbp1',
#        'sf1', 'sf3b1', 'simc1', 'smc3','srsf2', 'stag2',
#        'stat3', 'suz12', 'tet2', 'tp53', 'u2af1', 'u2af2',
#        'wt1', 'zrsr2']]

# percentages_genes_mds = percentages_mds[[
#         'apc',
#        'asxl1', 'bcor', 'bcorl1', 'braf', 'c7orf55', 'cbl', 'ccdc42b', 'cdh23',
#        'cdkn2a', 'cebpa', 'cftr', 'csf1r', 'cux1', 'ddx41', 'ddx54', 'dhx29',
#        'dnmt3a', 'eed', 'erbb4', 'etv6', 'ezh2', 'flt3', 'gata2',
#        'gli1', 'gli2', 'gnb1', 'gpr98', 'idh1', 'idh2', 'irf4', 'jak2',
#        'jak3', 'kdm5a', 'kdm6a', 'kit', 'kras', 'luc7l2',
#        'mecom', 'med12', 'mll', 'mpl', 'nf1',
#        'notch1', 'npm1', 'nras', 'ogt', 'phf6', 'prpf40b', 'prpf8',
#        'ptch1', 'pten', 'ptpn11', 'rad21', 'rnf25', 'runx1', 'setbp1',
#        'sf1', 'sf3b1', 'simc1', 'smc3','srsf2', 'stag2',
#        'stat3', 'suz12', 'tet2', 'tp53', 'u2af1', 'u2af2',
#        'wt1', 'zrsr2']]

# percentages_genes_cmml = percentages_cmml[[
#         'apc',
#        'asxl1', 'bcor', 'bcorl1', 'braf', 'c7orf55', 'cbl', 'ccdc42b', 'cdh23',
#        'cdkn2a', 'cebpa', 'cftr', 'csf1r', 'cux1', 'ddx41', 'ddx54', 'dhx29',
#        'dnmt3a', 'eed', 'erbb4', 'etv6', 'ezh2', 'flt3', 'gata2',
#        'gli1', 'gli2', 'gnb1', 'gpr98', 'idh1', 'idh2', 'irf4', 'jak2',
#        'jak3', 'kdm5a', 'kdm6a', 'kit', 'kras', 'luc7l2',
#        'mecom', 'med12', 'mll', 'mpl', 'nf1',
#        'notch1', 'npm1', 'nras', 'ogt', 'phf6', 'prpf40b', 'prpf8',
#        'ptch1', 'pten', 'ptpn11', 'rad21', 'rnf25', 'runx1', 'setbp1',
#        'sf1', 'sf3b1', 'simc1', 'smc3','srsf2', 'stag2',
#        'stat3', 'suz12', 'tet2', 'tp53', 'u2af1', 'u2af2',
#        'wt1', 'zrsr2']]

# # print('percentages_genes_mds_cmml\n', percentages_genes_mds_cmml.sort_values(ascending=False))

# # print('percentages_genes_mds\n', percentages_genes_mds.sort_values(ascending=False))

# # print('percentages_genes_cmml\n', percentages_genes_cmml.sort_values(ascending=False))


# df=pd.DataFrame({'MDS_CMML':percentages_genes_mds_cmml, 'MDS':percentages_genes_mds, 'CMML':percentages_genes_cmml})
# # writer = pd.ExcelWriter('gene frequencies.xlsx')
# # df.to_excel(writer,'gene frequencies')
# # writer.save()

# plt.rcParams['font.size'] = 16
# plt.rcParams['figure.figsize'] = (40, 9)
# df.plot(kind='bar', stacked=False)
# pl.savefig('Mutational Frequencies.png', dpi=400, transparent=False, bbox_inches='tight')
# pl.title('Mutational Frequencies for Whole Dataset, MDS, and CMML')
# pl.show()
# pl.close()


# # Remove BM blast column (further censor BM data)
# # and WBC (avoid colinearity, prefer specific lineages)

# MDS_data = MDS_data[MDS_data.columns.drop(list(MDS_data.filter(items=[
#                                                                       'percent bm blasts',
#                                                                       'wbc'

#                                                                      ])))]


# # make mutations only database
# MDS_data_mutations_only =  MDS_data[[
#         'apc',
#        'asxl1', 'bcor', 'bcorl1', 'braf', 'c7orf55', 'cbl', 'ccdc42b', 'cdh23',
#        'cdkn2a', 'cebpa', 'cftr', 'csf1r', 'cux1', 'ddx41', 'ddx54', 'dhx29',
#        'dnmt3a', 'eed', 'erbb4', 'etv6', 'ezh2', 'flt3', 'gata2',
#        'gli1', 'gli2', 'gnb1', 'gpr98', 'idh1', 'idh2', 'irf4', 'jak2',
#        'jak3', 'kdm5a', 'kdm6a', 'kit', 'kras', 'luc7l2',
#        'mecom', 'med12', 'mll', 'mpl', 'nf1',
#        'notch1', 'npm1', 'nras', 'ogt', 'phf6', 'prpf40b', 'prpf8',
#        'ptch1', 'pten', 'ptpn11', 'rad21', 'rnf25', 'runx1', 'setbp1',
#        'sf1', 'sf3b1', 'simc1', 'smc3','srsf2', 'stag2',
#        'stat3', 'suz12', 'tet2', 'tp53', 'u2af1', 'u2af2',
#        'wt1', 'zrsr2',
#        #'total number of mutations'
#           ]]

# features_mut = MDS_data_mutations_only


# # make clinical features only database

# MDS_data_clinical_only = MDS_data[MDS_data.columns.drop(list(MDS_data.filter(items=[
#        'apc',
#        'asxl1', 'bcor', 'bcorl1', 'braf', 'c7orf55', 'cbl', 'ccdc42b', 'cdh23',
#        'cdkn2a', 'cebpa', 'cftr', 'csf1r', 'cux1', 'ddx41', 'ddx54', 'dhx29',
#        'dnmt3a', 'eed', 'erbb4', 'etv6', 'ezh2', 'flt3', 'gata2',
#        'gli1', 'gli2', 'gnb1', 'gpr98', 'idh1', 'idh2', 'irf4', 'jak2',
#        'jak3', 'kdm5a', 'kdm6a', 'kit', 'kras', 'luc7l2',
#        'mecom', 'med12', 'mll', 'mpl', 'nf1',
#        'notch1', 'npm1', 'nras', 'ogt', 'phf6', 'prpf40b', 'prpf8',
#        'ptch1', 'pten', 'ptpn11', 'rad21', 'rnf25', 'runx1', 'setbp1',
#        'sf1', 'sf3b1', 'simc1', 'smc3','srsf2', 'stag2',
#        'stat3', 'suz12', 'tet2', 'tp53', 'u2af1', 'u2af2',
#        'wt1', 'zrsr2',
#        'total number of mutations',
#        'who category'
#                                                          ])))]

# features_clin = MDS_data_clinical_only


# # selected features - top features based on previous SHAP analysis

# features = MDS_data[[
#                     'abs mono', 'abs lym', 'tet2', 'abs neut', 'asxl1', 'sf3b1', 'hgb', 'total number of mutations',
#                     'abs eos', 'age', 'plt', 'runx1', 'gender', 'abs bas',
#                     'nras', 'cbl',  'u2af1', 'stag2', 'dnmt3a',
#                      'tp53','ezh2', 'srsf2', 'zrsr2'
#                     ]]


# # fix capitalization for manuscript and presentation figures

# features.rename(columns={
#                   'abs mono' : 'Absolute Monocyte Count', 'abs lym' : 'Absolute Lymphocyte Count', 'tet2' : 'TET2',
#                   'abs neut' : 'Absolute Neutrophil Count', 'asxl1' : 'ASXL1',
#                   'sf3b1' : 'SF3B1', 'hgb' : 'Hemoglobin', 'total number of mutations' : 'Total Number of Mutations',
#                   'abs eos' : 'Absolute Eosinophil Count', 'age': 'Age', 'plt' : 'Platelet Count', 'splenomegaly' : 'Splenomegaly',
#                   'runx1' : 'RUNX1',
#                   'gender' : 'Gender', 'abs bas' : 'Absolute Basophil Count',
#                   'nras' : 'NRAS', 'cbl' : 'CBL',  'u2af1' : 'U2AF1', 'stag2' : 'STAG2', 'dnmt3a' : 'DNMT3A',
#                   'tp53' : 'TP53','ezh2' : 'EZH2', 'srsf2' : 'SRSF2', 'zrsr2' : 'ZRSR2'
#                   }, inplace=True)

# features_mut.rename(columns={
#                   'abs mono' : 'Absolute Monocyte Count', 'abs lym' : 'Absolute Lymphocyte Count', 'tet2' : 'TET2',
#                   'abs neut' : 'Absolute Neutrophil Count', 'asxl1' : 'ASXL1',
#                   'sf3b1' : 'SF3B1', 'hgb' : 'Hemoglobin', 'total number of mutations' : 'Total Number of Mutations',
#                   'abs eos' : 'Absolute Eosinophil Count', 'age': 'Age', 'plt' : 'Platelet Count', 'splenomegaly' : 'Splenomegaly',
#                   'runx1' : 'RUNX1',
#                   'gender' : 'Gender', 'abs bas' : 'Absolute Basophil Count',
#                   'nras' : 'NRAS', 'cbl' : 'CBL',  'u2af1' : 'U2AF1', 'stag2' : 'STAG2', 'dnmt3a' : 'DNMT3A',
#                   'tp53' : 'TP53','ezh2' : 'EZH2', 'srsf2' : 'SRSF2', 'zrsr2' : 'ZRSR2'
#                   }, inplace=True)

# features_clin.rename(columns={
#                   'abs mono' : 'Absolute Monocyte Count', 'abs lym' : 'Absolute Lymphocyte Count', 'tet2' : 'TET2',
#                   'abs neut' : 'Absolute Neutrophil Count', 'asxl1' : 'ASXL1',
#                   'sf3b1' : 'SF3B1', 'hgb' : 'Hemoglobin', 'total number of mutations' : 'Total Number of Mutations',
#                   'abs eos' : 'Absolute Eosinophil Count', 'age': 'Age', 'plt' : 'Platelet Count', 'splenomegaly' : 'Splenomegaly',
#                   'runx1' : 'RUNX1',
#                   'gender' : 'Gender', 'abs bas' : 'Absolute Basophil Count',
#                   'nras' : 'NRAS', 'cbl' : 'CBL',  'u2af1' : 'U2AF1', 'stag2' : 'STAG2', 'dnmt3a' : 'DNMT3A',
#                   'tp53' : 'TP53','ezh2' : 'EZH2', 'srsf2' : 'SRSF2', 'zrsr2' : 'ZRSR2'
#                   }, inplace=True)


# final_combo = MDS_data

# writer = pd.CSVWriter("01 CCF_MLL_MDS_CMML.xlsx")
# final_combo.to_excel(writer,'Sheet1')
# writer.save()
