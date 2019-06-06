import os

import numpy as np
import pandas as pd

from cbh import config

pd.options.mode.chained_assignment = None  # default='warn'
print("Loading", os.path.basename(__file__))
### read in data ###
mds = pd.read_excel(config.RAW_MLL_MDS_FILE)

col_list = sorted(list(mds))
txt_file = "zzz_mds_raw.txt"

with open(config.DOCS_DIR/txt_file, "w") as f:
    f.write("MDS\n\n")
    for item in col_list:
        f.write("%s\n" % item)



# clean up the column names and data en masse
dataframes = [mds]
for df in dataframes:
    df.columns = df.columns.str.strip().str.lower().str.replace("  ", " ")
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)

mds.rename(index=str, columns={"mll_id": "id", "who": "diagnosis", 'anc' : 'abs neut', "hb": "hgb"}, inplace=True)

# #XX and XY to male and female
mds.loc[mds['cytogenetic'].str.contains('XY',case=False), 'gender'] = 'male'
mds.loc[mds['cytogenetic'].str.contains('X,-Y',case=False), 'gender'] = 'male'
mds.loc[mds['cytogenetic'].str.contains('XX',case=False), 'gender'] = 'female'
mds.loc[mds['cytogenetic'].str.contains('X,-X',case=False), 'gender'] = 'female'

# print(mds.shape)

# mds = mds[mds.columns.drop(list(mds.filter(regex='vaf')))]
# mds = mds[mds.columns.drop(list(mds.filter(items=['cellularity',
#                                               #'percent bm blasts',
#                                               'cytogenetic points - ipss-r',
#                                               'megakaryocytes', 'erythropoiesis','granulopoiesis'
#                                                                      ])))]

# print(mds.shape)

# #drop redundancies and other unwanted columns
# print(mds.shape)
mds = mds[mds.columns.drop(list(mds.filter(regex="vaf")))]
mds = mds[mds.columns.drop(list(mds.filter(items=['race',
                                              #'wbc',
                                            #   'mono %', 'lym %','neut %', 'eos %', 'bas %',
                                              'cytogenetic'
                                               ])))]
# print(mds.shape)

#fix scale to match CCF dataset
mds['abs lym'] = mds['abs lym']/1000
mds['abs mono'] = mds['abs mono']/1000
mds['abs eos'] = mds['abs eos']/1000
mds['abs bas'] = mds['abs bas']/1000


col_list = sorted(list(mds))
txt_file = "zz_mds.txt"
# print(col_list)
with open(config.DOCS_DIR/txt_file, "w") as f:
    f.write("MDS\n\n")
    for item in col_list:
        f.write("%s\n" % item)


mds.to_hdf(config.RAW_DATA_FILE_H5, key="mds", mode="a", format="table")
