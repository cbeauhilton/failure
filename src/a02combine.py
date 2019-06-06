import contextlib
import os
import warnings
from functools import partial, reduce

import h5py
import numpy as np
import pandas as pd

from cbh import config

warnings.simplefilter(action='ignore', category=UserWarning)
print("Loading", os.path.basename(__file__))


# f = h5py.File(config.RAW_DATA_FILE_H5, "r")
# keylist = list(f.keys())
# print(keylist)

ccf = pd.read_hdf(config.RAW_DATA_FILE_H5, key="ccf")
cmml = pd.read_hdf(config.RAW_DATA_FILE_H5, key="cmml")
mds = pd.read_hdf(config.RAW_DATA_FILE_H5, key="mds")
mpn = pd.read_hdf(config.RAW_DATA_FILE_H5, key="mpn")

# for df in [ccf, cmml, mds, mpn]:
#     # print()
#     print(df.select_dtypes(include=['object']))

filename = "zz_all.txt"
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


# def diff(list1, list2):
#     return list(set(list1).symmetric_difference(set(list2)))

# print(diff(sorted(list(ccf)), sorted(list(cmml))))

df0 = pd.DataFrame({"CCF": sorted(list(ccf))})
df1 = pd.DataFrame({"CMML": sorted(list(cmml))})
df2 = pd.DataFrame({"MDS": sorted(list(mds))})
df3 = pd.DataFrame({"MPN": sorted(list(mpn))})
df = pd.concat([df0, df1, df2, df3], axis=1)
# print(df)
df.to_csv(config.DOCS_DIR / "zz_combo.csv")

# select only the columns that match between all sets
all_cols = sorted(list(set(ccf) & set(cmml) & set(mds) & set(mpn)))
# print(all_cols)
# print(len(all_cols))

dfs = [ccf[all_cols], cmml[all_cols], mds[all_cols], mpn[all_cols]]
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
    if (x.dtype == "object")
    else x
)

data.replace({"diagnosis": {"raeb-1": "mds-eb", "raeb-2": "mds-eb"}}, inplace=True)
data.replace(
    {"diagnosis": {"rars": "mds-rs", "rars-t": "mds-mpn-rs-t", "rcmd-rs": "mds-rs"}},
    inplace=True,
)
data.replace({"diagnosis": {"cmml-1": "cmml", "cmml-2": "cmml"}}, inplace=True)
data.replace({"diagnosis": {"rcud": "mds-sld", "ra": "mds-sld"}}, inplace=True)
data.replace({"diagnosis": {"rcmd": "mds-mld"}}, inplace=True)
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


# get rid of rare diagnoses
col = "diagnosis"
n = 64
data = data[data.groupby(col)[col].transform("count").ge(n)]

# print(data.diagnosis.value_counts(dropna=False))
# print(len(data.diagnosis.notnull()))
# print(data.count())


data.columns = [col.replace("%", "percent") for col in data.columns]
data.columns = [col.replace(" ", "_") for col in data.columns]

cats = [
    "asxl1",
    "bcor",
    "cbl",
    "cebpa",
    "dnmt3a",
    "etv6",
    "ezh2",
    "gata2",
    "gender",
    # "id",
    "idh1",
    "idh2",
    "jak2",
    "kdm6a",
    "kit",
    "kras",
    "nf1",
    "npm1",
    "nras",
    "phf6",
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
data["id"] = data["id"].astype("category")
data[cats] = data[cats].astype("category")
one_hot = pd.get_dummies(data[cats],dummy_na=True)
# Drop columns now encoded
data = data.drop(cats, axis=1)
# Join the encoded df
data = data.join(one_hot)

# print(list(data))
data.to_hdf(config.RAW_DATA_FILE_H5, key="data", mode="a", format="table")

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
