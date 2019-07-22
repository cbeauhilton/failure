from cbh import config
import os
import pandas as pd
import numpy as np
import collections
import operator
import re

df = pd.read_hdf(config.RAW_DATA_FILE_H5, key="data")
y = pd.read_hdf(config.RAW_DATA_FILE_H5, key="y")
# X = pd.read_hdf(config.RAW_DATA_FILE_H5, key="X")

###############################################################################
#        ___
#   ____/ (_)___ _____ ______
#  / __  / / __ `/ __ `/ ___/
# / /_/ / / /_/ / /_/ (__  )
# \__,_/_/\__,_/\__, /____/
#              /____/
###############################################################################

df['institution'] = np.where(df['id'].str.contains("mll_", case=False, na=False), 'MLL', 'CCF')
# print(df[['institution', "id"]])
institutions = np.unique(df["institution"])
# print(institutions)
friends = {}
for place in institutions:
    friends[place] = len(df[df["institution"] == place])

# print(friends)

classes = np.unique(y) # retrieve all class names
CLASSES = [x.upper() for x in classes] # make uppercase version
Classes = [x.title() for x in classes] # make titlecase version

# Start the class_sizes dictionary with the length of the entire cohort
class_sizes = {"cohort_size": len(df)}

# append a key, value pair with the diagnosis and number of pts
for diag in classes:
    size = len(df.loc[df["diagnosis"]== f"{diag}"])
    class_sizes[f"{diag}"] = size

# order the dictionary by descending number of pts
sorted_class_sizes = collections.OrderedDict(sorted(class_sizes.items(), key=lambda t: t[1], reverse=True))

# pull the first value (cohort size) into one variable,
# all of the middle values into a second variable, 
# and the last value into its own variable
a, *b, c = sorted_class_sizes
# print(a) ; print(b) ; print(c)

count_clause_1 = []
for i in b:
    sent = f"{class_sizes[i]} had {i.upper()}, "
    count_clause_1.append(sent)

count_clause_0 = f"Of {class_sizes[a]} pts included, "
count_clause_1 = ''.join(count_clause_1)
count_clause_2 = f"and {class_sizes[c]} had {c.upper()}. "
# print(count_clause_0) ; print(count_clause_1) ; print(count_clause_2)
count_sent = count_clause_0 + count_clause_1 + count_clause_2
# print(count_sent)


###############################################################################
#                                      __  _ __
#     ____  ___  _____________  ____  / /_(_) /__  _____
#    / __ \/ _ \/ ___/ ___/ _ \/ __ \/ __/ / / _ \/ ___/
#   / /_/ /  __/ /  / /__/  __/ / / / /_/ / /  __(__  )
#  / .___/\___/_/   \___/\___/_/ /_/\__/_/_/\___/____/
# /_/
###############################################################################


# Percentiles for all numeric columns for overall cohort:
percentile_dict = {}
for col in list(df.select_dtypes(include=[np.number]).columns.values):    
    percentile_dict[f"{col}_min"] = df[f"{col}"].min() 
    percentile_dict[f"{col}_mean"] = df[f"{col}"].mean()
    percentile_dict[f"{col}_q1"] = df[f"{col}"].quantile(0.25)
    percentile_dict[f"{col}_median"] = df[f"{col}"].median()
    percentile_dict[f"{col}_q3"] = df[f"{col}"].quantile(0.75)
    percentile_dict[f"{col}_max"]= df[f"{col}"].max()


###############################################################################
#                              _   __
#    ____ ____  ____  ___     (_)_/_/
#   / __ `/ _ \/ __ \/ _ \     _/_/
#  / /_/ /  __/ / / /  __/   _/_/_
#  \__, /\___/_/ /_/\___/   /_/ (_)
# /____/
###############################################################################


# Ordered gene percentages
def get_ordered_gene_percents(firstNpairs, dict_of_strings, percentile_dict=percentile_dict, target="all_genes"):
    # grab all of the mean gene findings
    cohort_genes_percents = dict([(key, value) for key, value in percentile_dict.items() if "_positive_mean" in key])
    for value in cohort_genes_percents:
        cohort_genes_percents[value] = cohort_genes_percents[value]*100

    # sort them by percentages
    sorted_cohort_genes_percents = collections.OrderedDict(sorted(cohort_genes_percents.items(), key=lambda t: t[1], reverse=True))
    # print(sorted_cohort_genes_percents)

    # grab first N pairs
    sorted_cohort_genes_firstNpairs = list(sorted_cohort_genes_percents.items())[:firstNpairs]

    all_genes = []
    for gene in range(len(sorted_cohort_genes_firstNpairs)):
        names = [i[0] for i in sorted_cohort_genes_firstNpairs]
        percents = [i[1] for i in sorted_cohort_genes_firstNpairs]
        name = names[gene]
        percent = percents[gene]
        name = re.sub('_positive_mean$', '', name)
        name = name.upper()
        pair = f"{name} ({percent:.1f}%),"
        # print(name)
        # print(percent)
        all_genes.append(pair)

    all_genes = ' '.join(all_genes)
    all_genes = all_genes[:-1]
    genes_percents = all_genes + "."
    # print(genes_percents)
    dict_of_strings[target] = genes_percents
    return genes_percents


gene_strings = {}
firstNpairs = 3

mutations_per_sample = {}
gene_cols = [col for col in list(df) if "_positive" in col]
df['mutation_num'] = df[gene_cols].sum(axis=1)
mutations_per_sample[f'cohort_min'] = df['mutation_num'].min()
mutations_per_sample[f'cohort_mean'] = df['mutation_num'].mean()
mutations_per_sample[f'cohort_median'] = df['mutation_num'].median()
mutations_per_sample[f'cohort_max'] = df['mutation_num'].max()

# all classes
get_ordered_gene_percents(firstNpairs, gene_strings, percentile_dict, target="The most commonly mutated genes in all pts were: ")
# per class
for classname in classes:
    percentile_dict[f'{classname}'] = {}
    df1 = df.loc[df["diagnosis"]== f"{classname}"]
    df1['mutation_num'] = df1[gene_cols].sum(axis=1)
    mutations_per_sample[f'{classname}_min'] = df1['mutation_num'].min()
    mutations_per_sample[f'{classname}_mean'] = df1['mutation_num'].mean()
    mutations_per_sample[f'{classname}_median'] = df1['mutation_num'].median()
    mutations_per_sample[f'{classname}_max'] = df1['mutation_num'].max()
    for col in list(df1.select_dtypes(include=[np.number]).columns.values):    
        percentile_dict[f'{classname}'][f"{col}_mean"] = df1[f"{col}"].mean()
    get_ordered_gene_percents(firstNpairs, gene_strings, percentile_dict=percentile_dict[f'{classname}'], target=f"In {classname.upper()}, they were: ")

# print(mutations_per_sample)
# print(percentile_dict)
# print(all_genes)
# print(gene_strings)
gene_percent_paragraph = []

for k,v in gene_strings.items():
    sent = k + v
    gene_percent_paragraph.append(''.join(sent))

gene_percent_paragraph = ' '.join(gene_percent_paragraph)
# print(gene_percent_paragraph)


###############################################################################
#                     __        __  _                  __ __
#    ____ ___  __  __/ /_____ _/ /_(_)___  ____     __/ // /_
#   / __ `__ \/ / / / __/ __ `/ __/ / __ \/ __ \   /_  _  __/
#  / / / / / / /_/ / /_/ /_/ / /_/ / /_/ / / / /  /_  _  __/
# /_/ /_/ /_/\__,_/\__/\__,_/\__/_/\____/_/ /_/    /_//_/
###############################################################################

mutation_num_sent = []

mutation_num_clause_0 = f"The median total number of mutations/sample was {mutations_per_sample['cohort_median']:.0f} (range {mutations_per_sample['cohort_min']}-{mutations_per_sample['cohort_max']}) for all pts" 
mutation_num_sent.append(mutation_num_clause_0)

for classname in classes:
    mini = mutations_per_sample[f"{classname}_min"]
    medi = mutations_per_sample[f"{classname}_median"]
    maxi = mutations_per_sample[f"{classname}_max"]
    clause = f"{medi:.0f} (range {mini}-{maxi}) for {classname.upper()}"
    mutation_num_sent.append(clause)

a, *b, c = mutation_num_sent
# print(b)
mutation_num_sent = a + ', ' + ', '.join(b) + " and " + c + '.'
# print(mutation_num_sent)

###############################################################################
#          _____ __
#    ____ / ___// /_  ____ _____
#   / __ \\__ \/ __ \/ __ `/ __ \
#  / / / /__/ / / / / /_/ / /_/ /
# /_/ /_/____/_/ /_/\__,_/ .___/
#                       /_/
###############################################################################


shap_vars = pd.read_csv(config.TABLES_DIR / "shap_df_pretty.csv")

nShaps = 20
shap_dict = {}
shap_class_clause_0 = f"These variables included: "
shap_class_clause_3 = f";"

for classname in classes:
    classname = classname.upper()
    shap_dict[f"{classname}_abbr"] = list(shap_vars[f"{classname}_abbr"][:nShaps])
    shap_dict[f"{classname}_full"] = list(shap_vars[f"{classname}"][:nShaps])
    
    shap_class_clause_1 = f"({classname}) "
    shap_class_clause_2 = shap_dict[f'{classname}_abbr']
    shap_class_clause_2 = ', '.join(shap_class_clause_2)
    shap_dict[f"{classname}_clause"] = shap_class_clause_1 + shap_class_clause_2 + shap_class_clause_3
    # print(shap_dict[f"{classname}_clause"])


shap_clauses = dict([(key, value) for key, value in shap_dict.items() if "_clause" in key])
shap_paragraph = []
for k,v in shap_clauses.items():
    sent = v
    shap_paragraph.append(''.join(sent))

shap_paragraph = ' '.join(shap_paragraph)
shap_paragraph = shap_paragraph[:-1]
# shap_paragraph = shap_paragraph + '.'
# print(shap_paragraph)


###############################################################################
#                         __     __
#    ____ ___  ____  ____/ /__  / /
#   / __ `__ \/ __ \/ __  / _ \/ /
#  / / / / / / /_/ / /_/ /  __/ /
# /_/ /_/ /_/\____/\__,_/\___/_/
###############################################################################

# model_performance = f"{}"

###############################################################################
#                                  __        ________
#   ___  ____ ________  __   _____/ /___  __/ __/ __/
#  / _ \/ __ `/ ___/ / / /  / ___/ __/ / / / /_/ /_
# /  __/ /_/ (__  ) /_/ /  (__  ) /_/ /_/ / __/ __/
# \___/\__,_/____/\__, /  /____/\__/\__,_/_/ /_/
#                /____/
###############################################################################


age_sent = f"The median age for the entire cohort was {percentile_dict['age_median']:.0f} years (range, {percentile_dict['age_min']:.0f}-{percentile_dict['age_max']:.0f}). "
gender_sent = f"{percentile_dict['gender_female_mean']*100:.0f}% were female. "
wbc_clause = f'The median white blood cell count (WBC) was {percentile_dict["wbc_median"]:.1f}x10^9/L (range, {percentile_dict["wbc_min"]:.2f}-{percentile_dict["wbc_max"]:.0f}), '
#  WBC IS WEIRD, SHOULD BE CLOSER TO: 176
amc_clause = f'absolute monocyte count (AMC) {percentile_dict["abs_mono_median"]:.2f}x10^9/L (range, {percentile_dict["abs_mono_min"]:.0f}-{percentile_dict["abs_mono_max"]:.0f}), '
alc_clause = f'absolute lymphocyte count (ALC) {percentile_dict["abs_lym_median"]:.2f}x10^9/L (range, {percentile_dict["abs_lym_min"]:.0f}-{percentile_dict["abs_lym_max"]:.0f}), '
anc_clause = f'absolute neutrophil count (ANC) {percentile_dict["abs_neut_median"]:.2f}x10^9/L (range, {percentile_dict["abs_neut_min"]:.0f}-{percentile_dict["abs_neut_max"]:.0f}), ' 
# ANC ALSO WEIRD, SHOULD BE CLOSER TO: 170
hgb_clause = f'and hemoglobin (Hgb) {percentile_dict["hgb_median"]:.2f} (range, {percentile_dict["hgb_min"]:.1f}-{percentile_dict["hgb_max"]:.1f}). '


###############################################################################
#    __            __
#   / /_____  ____/ /___
#  / __/ __ \/ __  / __ \
# / /_/ /_/ / /_/ / /_/ /
# \__/\____/\__,_/\____/
###############################################################################

# order: MDS, ICUS, CCUS, CMML, MDS/MPN, PV, ET, PMF 



###############################################################################
#     __               __                                    __
#    / /_  ____ ______/ /______ __________  __  ______  ____/ /
#   / __ \/ __ `/ ___/ //_/ __ `/ ___/ __ \/ / / / __ \/ __  /
#  / /_/ / /_/ / /__/ ,< / /_/ / /  / /_/ / /_/ / / / / /_/ /
# /_.___/\__,_/\___/_/|_|\__, /_/   \____/\__,_/_/ /_/\__,_/
#                       /____/
###############################################################################


background = f"""
Background

Myelodysplastic syndromes (MDS) and other myeloid neoplasms \
are mainly diagnosed based on morphological changes in the bone marrow. \
The diagnosis can be challenging in patients (pts) with pancytopenia with minimal dysplasia, \
and is subject to inter-observer variability. \
Somatic mutations can be identified in all myeloid neoplasms \
but no gene or set of genes are diagnostic for each disease phenotype. \

We developed a geno-clinical model that uses \
mutational data, peripheral blood values, and clinical variables \
to distinguish between several bone marrow disorders that include: \
MDS, \
idiopathic cytopenia of indeterminate significance (ICUS), clonal cytopenia of indeterminate significance (CCUS), \
MDS/MPN overlaps including chronic myelomonocytic leukemia (CMML), \
and myeloproliferative neoplasms such as \
polycythemia vera (PV), essential thrombocythemia (ET), and myelofibrosis (PMF).

"""

###############################################################################
#                    __  __              __
#    ____ ___  ___  / /_/ /_  ____  ____/ /____
#   / __ `__ \/ _ \/ __/ __ \/ __ \/ __  / ___/
#  / / / / / /  __/ /_/ / / / /_/ / /_/ (__  )
# /_/ /_/ /_/\___/\__/_/ /_/\____/\__,_/____/
###############################################################################  
# {CLASSES[0]} 
methods = f"""
Methods

We combined genomic and clinical data from {len(df)} pts \
treated at our institution ({friends["CCF"]}) and the Munich Leukemia Laboratory ({friends["MLL"]}). \
Pts were diagnosed with {", ".join(CLASSES[:-1])} and {CLASSES[-1]} according to 2008 WHO criteria. \
Diagnosis was confirmed by independent hematopathologists not associated with the study. \
A genomic panel of {len(gene_cols)} genes commonly mutated in myeloid malignancies was included. \
The initial cohort was randomly (computer generated) divided into \
learner (80%) and validation (20%) cohorts. \
Multiple machine learning algorithms were applied to predict the phenotype. \
Feature extraction algorithms were used to extract genomic/clinical variables \
that impacted the algorithm decision and to visualize the impact of each variable on phenotype. \
Prediction performance was evaluated according to \
the area under the curve of the receiver operator characteristic (ROC-AUC) and confusion/accuracy matrices.

"""

###############################################################################
#                          ____
#    ________  _______  __/ / /______
#   / ___/ _ \/ ___/ / / / / __/ ___/
#  / /  /  __(__  ) /_/ / / /_(__  )
# /_/   \___/____/\__,_/_/\__/____/
###############################################################################

results = f"""
Results

{count_sent}{age_sent}{gender_sent}{wbc_clause}{amc_clause}{alc_clause}{anc_clause}{hgb_clause}

{gene_percent_paragraph}

{mutation_num_sent}


A set of {len(list(df))} genomic/clinical variables were evaluated \
and several feature extraction algorithms were used to identify the \
variables that have the most significant impact on the algorithm's decision. \
These variables included: \
{shap_paragraph} (Figure 1).

When applying the model to the validation cohort, \
the ROC-AUC was .98 with an accuracy of 94%, \
with other statistical values as follows: \
specificities CMML 93%, MDS 96%; \
sensitivities CMML 96%, MDS 93%; \
positive predictive values CMML 84%, MDS 98%; \
negative predictive values CMML 98%, MDS 84%.

Individual pt data can also be entered into the model, \
with a probability of whether the diagnosis is MDS vs. CMML provided along with the impact of each variable on the decision, as shown in Figure 1.

When the analysis was restricted to mutations only, \
the accuracy of the model dropped dramatically (77%, ROC-AUC .85).

"""



###############################################################################
#                          __           _
#   _________  ____  _____/ /_  _______(_)___  ____  _____
#  / ___/ __ \/ __ \/ ___/ / / / / ___/ / __ \/ __ \/ ___/
# / /__/ /_/ / / / / /__/ / /_/ (__  ) / /_/ / / / (__  )
# \___/\____/_/ /_/\___/_/\__,_/____/_/\____/_/ /_/____/
###############################################################################


conclusions = f"""
Conclusions

We propose a novel approach using interpretable, individualized modeling \
to predict myeloid neoplasm phenotypes based on genomic and clinical data \
without the need for bone marrow biopsy data. \
This approach can aid clinicians and hematopathologists when encountering \
pts with cytopenias and suspicion for these disorders. \
The model also provides feature attributions \
that allow for quantitative understanding \
of the complex interplay among genotypes, clinical variables, and phenotypes.


"""



###############################################################################
#    _________ __   _____
#   / ___/ __ `/ | / / _ \
#  (__  ) /_/ /| |/ /  __/
# /____/\__,_/ |___/\___/
###############################################################################




abstract = background + methods + results + conclusions

abstract = re.sub('MDS_MPN', 'MDS/MPN', abstract)

chars_no_spaces = len(''.join(abstract.split()))
curr_char = f"Number of characters in abstract: {chars_no_spaces}. \n"
ASH_limit = "ASH character limit is 3800. \n"

if chars_no_spaces > 3800:
    editing = f"You are over by {chars_no_spaces - 3800} non-whitespace characters.\n"
elif chars_no_spaces < 3800:
    editing = f"You are under by {3800 - chars_no_spaces} non-whitespace characters.\n"
else:
    editing = f"Nailed it. 3800 = {chars_no_spaces}.\n"

char_report = curr_char + ASH_limit + editing + "\n" + "===" + "\n"

combined = char_report + abstract
# print(combined)

# txt for copy-pasta, spot-checking
out_text_file = config.TEXT_DIR / "00_abstract_conference.txt"
# rtf for auto-include in Word docs
out_rtf_file = config.TEXT_DIR / "00_abstract_conference.rtf"
# tex for awesome
out_file_latex = config.TEX_SECTIONS_DIR / "00_abstract_conference.tex"

# Define file...
if not os.path.exists(config.TEX_SECTIONS_DIR):
    print("Making folder called", config.TEX_SECTIONS_DIR)
    os.makedirs(config.TEX_SECTIONS_DIR)

if not os.path.exists(config.TEXT_DIR):
    print("Making folder called", config.TEXT_DIR)
    os.makedirs(config.TEXT_DIR)

# ...and save.
with open(out_text_file, "w") as text_file:
    print("\n")
    print(out_text_file)
    print(combined, file=text_file)

try:
    with open(out_rtf_file, "w") as text_file:
        print(out_rtf_file)
        print(combined, file=text_file)
except:
    print("RTF not saved.")

# and make a LaTeX-friendly version
with open(out_text_file, "r") as file:
    filedata = file.read()

# Replace the percentage symbols with escaped versions
filedata = filedata.replace("%", "\%")

# convert superscripts
# grabs literal caret symbol ^, 
# \s* means "any number of whitespaces including zero," 
# (\d*) grabs any number of digits and stores them, 
# then the \1 inserts those numbers into the right place.
filedata = re.sub(r'\^\s*(\d*)' , r'\\textsuperscript{\1}',filedata)

# remove everything up to the three equal signs
filedata = re.sub(r'(.|\n)*===\n*', "", filedata)
# print(filedata)
filedata = filedata.replace("Background", "\subsection{Background}%")
filedata = filedata.replace("Methods", "\subsection{Methods}%")
filedata = filedata.replace("Results", "\subsection{Results}%")
filedata = filedata.replace("Conclusions", "\subsection{Conclusions}%")
section_header = "\section{Abstract}%"
# Write the file
with open(out_file_latex, "w") as file:
    print(out_file_latex)
    file.write(section_header)
    file.write(filedata)








### Plt is missing ###
# and platelet (Plt) count {percentile_dict["plt_median"]:.2f}x10^3/L \
# (range, {percentile_dict["plt_min"]:.0f}-{percentile_dict["plt_max"]:.0f}),
# 111x10^3/mL (range 2-1491). \

# The most commonly mutated genes in all pts were: {all_genes} \
# TET2 (33%), ASXL1 (26%), SF3B1 (21%), SRSF2 (16%), \
# RUNX1 (12%), DNMT3A(10%), CBL (7%), U2AF1 (7%), \
# STAG2 (6%), EZH2 (6%), ZRSR2 (6%), NRAS (6%). \


# The most commonly mutated genes in all pts were: {all_genes} \

# In CMML, they were: \
# TET2 (51%), ASXL1 (43%), SRSF2(25%), RUNX1 (18%), \
# CBL (16%), KRAS (12%), NRAS (11%), EZH2(9%), \
# JAK2 (6%), U2AF1 (5%), SF3B1 (4%), and DNMT3A (3%). \
# In MDS, they were: \
# TET2 (27%), SF3B1 (24%), ASXL1 (21%), SRSF2(13%), \
# DNMT3A (12%), RUNX1 (10%), STAG2 (8%), U2AF1 (8%), \
# ZRSR2 (7%), TP53 (7%), BCOR (5%), and EZH2 (5%).\

# The median total number of mutations/sample \
# was 2 (range 0-27) for all pts, 2 (range 0-8) for CMML, and 2 (range 0-27) for MDS.

# AMC, ALC, TET2, ANC, ASXL1, SF3B1, Hgb, \
# number of mutations/sample, AEC, age, Plt, splenomegaly, \
# RUNX1, NRAS, CBL, U2AF1, STAG2, DNMT3A, \
# TP53, EZH2, SRSF2, and ZRSR2 