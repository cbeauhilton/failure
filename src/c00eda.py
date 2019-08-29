import os
import shutil
import sys
import traceback  # for error handling

import missingno as msno
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import modality
import numpy as np
import pandas as pd
from pandas_summary import DataFrameSummary
from scipy import stats
from tableone import TableOne
import cowsay

from cbh import config
from cbh.exhandler import exhandler


data = pd.read_hdf(config.RAW_DATA_FILE_H5, key="data")
X = pd.read_hdf(config.RAW_DATA_FILE_H5, key="X")
y = pd.read_hdf(config.RAW_DATA_FILE_H5, key="y")
# X = pd.read_hdf(config.RAW_DATA_FILE_H5, key="X")

df = X.copy()
df = df[df.columns.drop(list(df.filter(regex="_vus")))]
# df = df[df.columns.drop(list(df.filter(regex="_negative")))]
# df = df[df.columns.drop(list(df.filter(regex="_positive")))]
df = df[df.columns.drop(list(df.filter(regex="_nan")))]
df = df.dropna(axis=1, how="all")
df = df.drop(["pt_num"], axis=1)
df["diagnosis"] = data["diagnosis"].astype("category")


os.chdir(f"{config.FIGURES_DIR}")
# save the initial base directory
base_dir = os.getcwd()

# make the top level directory, where we'll put everything
# the `-p` option makes it not throw an error if the directory already exists
if not os.path.exists("exploratory_data_analysis"):
    os.makedirs("exploratory_data_analysis")

# move into that directory
os.chdir("exploratory_data_analysis")

try:
    os.remove("exceptions.txt")
except OSError:
    pass


if not os.path.exists("missingness"):
    os.makedirs("missingness")

fig = plt.Figure()
msno.matrix(df)
fig = plt.gcf()
fig.savefig('missingness/missing_matrix.pdf', format='pdf', bbox_inches='tight')
plt.close()

fig = plt.Figure()
msno.bar(df)
fig = plt.gcf()
fig.savefig('missingness/missing_bar.pdf', format='pdf', bbox_inches='tight')
plt.close()

fig = plt.Figure()
msno.heatmap(df)
fig = plt.gcf()
fig.savefig('missingness/missing_heatmap.pdf', format='pdf', bbox_inches='tight')
plt.close()

fig = plt.Figure()
msno.dendrogram(df)
fig = plt.gcf()
fig.savefig('missingness/missing_dendrogram.pdf', format='pdf', bbox_inches='tight')
plt.close()


if not os.path.exists("data_summaries"):
    os.makedirs("data_summaries")
data_describe = pd.DataFrame(df.describe(include="all"))
data_describe.to_csv("data_summaries/data_describe.csv")

dfs = DataFrameSummary(df)
columns_stats = pd.DataFrame(dfs.columns_stats)
columns_stats.to_csv("data_summaries/data_summary.csv")

data_summary = pd.DataFrame()
for col in list(df):
    data_summary[col] = dfs[col]
data_summary.to_csv("data_summaries/data_summary_extended.csv")

columns = list(df)

df_cats = df.select_dtypes(include="category")
groupby_df = df_cats.ix[:, df_cats.apply(lambda x: x.nunique()) <= 10]
groupbys = list(groupby_df)
# print(groupbys)
groupbys = list(set(groupbys) & set(columns))

categorical = list(df.select_dtypes(include="category"))
genes = config.GENE_COLS
mystring = "_positive"
gene_cols = [gene + mystring for gene in genes]
categorical = categorical + gene_cols + ['gender_female']
# print(categorical)
categorical = list(set(categorical) & set(columns))
print(categorical)

nonnormal = []
alpha = 0.05
for x in list(df):
    try:
        stat, p = stats.normaltest(df[x].values, nan_policy="omit")  # omit missing

        if p > alpha:
            # uncomment out these print statements for more verbose output
            #   print(f"{x} appears to fit a Gaussian distribution")
            #   print(f"p={p} \n")
            print("")

        else:
            nonnormal.append(x)
        #   print(f"{x} appears to be non-normally distributed")
        #   print(f"p={p} \n")

    except:
        # print(f"Could not determine normality for {x} \n")
        # print("")
        pass
        
nonnormal = list(set(nonnormal) & set(columns))

labels = {}
decimals = {}
# labels = {"TIBC": "Total Iron Binding Capacity", "TS": "Transferrin Saturation"}
# decimals = {"Age": 0, 'Sedimentation rate': 0, 'Diastolic BP': 0}

if not os.path.exists("tables_one"):
    os.makedirs("tables_one")


try:
    # make a non-grouped version first:
    nongrouped = TableOne(
        df,  # the data we will pass in
        #      groupby=groupby, # if we want to stratify the table (say, by tx group)
        columns=columns,  # which variables of `data` do we want to analyze?
        # if we want all of them, leave `columns` blank
        categorical=categorical,  # which variables are categorical?
        nonnormal=nonnormal,  # which variables are non-normally distributed?
        labels=labels,  # do we want to give new names to the variables?
        label_suffix=True,  # Add suffix to label (e.g. "mean (SD); median [Q1,Q3], n (%)")
        decimals=decimals,  # how many decimal places do we want to report for each variable?
        isnull=True,  # do we want to print how many values are missing of a given variable?
        #      pval=True, # do we want to print p-values?,
        #      pval_adjust='bonferroni', # do we want to adjust p-values for multiple testing? If so, how?
        #      ddof=1, # how many degrees of freedom for standard deviation calculations?
        #      sort=False, # do we want to sort the columns alphabetically, or by our own order?
        #      limit=5, # do we want to limit the report to the top N most frequent categories?
        remarks=True,  # do we want TableOne to tell us about our stats and any problems?
    )

    nongrouped.to_latex(f"tables_one/tableone_nongroupby.tex")
    nongrouped.to_html(f"tables_one/tableone_nongroupby.html")
    nongrouped.to_csv(f"tables_one/tableone_nongroupby.csv")

    # pull out the overall column as a list to add on to the other CSVs later
    # unfortunately, I do not know how to do this before generating the HTML or Tex
    get_overall = pd.read_csv(f"tables_one/tableone_nongroupby.csv")
    overall = list(get_overall["overall"])
    a = ["Overall", ""]
    overall = a + overall
    # print(overall)

    # Here's some secret sauce for exploratory data analysis:
    # make a loop to output a new TableOne for every category

    for groupby in groupbys:

        mytable = TableOne(
            df,  # the data we will pass in
            groupby=groupby,  # if we want to stratify the table (say, by tx group)
            columns=columns,  # which variables of `data` do we want to analyze?
            # if we want all of them, leave `columns` blank
            categorical=categorical,  # which variables are categorical?
            nonnormal=nonnormal,  # which variables are non-normally distributed?
            labels=labels,  # do we want to give new names to the variables?
            label_suffix=True,  # Add suffix to label (e.g. "mean (SD); median [Q1,Q3], n (%)")
            decimals=decimals,  # how many decimal places do we want to report for each variable?
            isnull=True,  # do we want to print how many values are missing of a given variable?
            #       pval=True, # do we want to print p-values?,
            #       pval_adjust='bonferroni', # do we want to adjust p-values for multiple testing? If so, how?
            #       ddof=1, # how many degrees of freedom for standard deviation calculations?
            #       sort=False, # do we want to sort the columns alphabetically, or by our own order?
            #       limit=5, # do we want to limit the report to the top N most frequent categories?
            remarks=True,  # do we want TableOne to tell us about our stats and any problems?
        )

        # take a look
        #   print(mytable)

        # Save table to LaTeX (for the nerds), HTML (for easy viewing), and CSV (for easy export)
        mytable.to_latex(f"tables_one/tableone_{groupby}.tex")
        mytable.to_html(f"tables_one/tableone_{groupby}.html")
        mytable.to_csv(f"tables_one/tableone_{groupby}.csv")

        try:
            table_with_overall = pd.read_csv(f"tables_one/tableone_{groupby}.csv")
            table_with_overall["Overall"] = overall
            table_with_overall.to_csv(f"tables_one/tableone_{groupby}.csv")
        except Exception as ex:
            exhandler(ex, module=os.path.basename(__file__))


    # print(nongrouped)
#   print(mytable)

except Exception as ex:
    exhandler(ex, module=os.path.basename(__file__))


plt.style.use("fivethirtyeight")

if not os.path.exists("histograms"):
    os.makedirs("histograms")


for col in list(df.select_dtypes(include=[np.number]).columns.values):
    try:
        #     print(col)
        plt.hist(df[col], edgecolor="black")
        plt.title(f"{col}")
        # find the median of each column
        median = df[col].median()
        # pick a color for the line
        color = "#fc4f30"
        # plot a red vertical line showing the median value
        plt.axvline(median, color=color, label="Median", linewidth=2)
        # I don't know why `tight_layout()` isn't the default. It's much prettier.
        plt.tight_layout()
        # Add the `.pdf` extension to get a pdf,
        # and add `transparent=True` to make the background transparent
        # (useful for making combined figures later)
        plt.savefig(f"histograms/{col}_hist.pdf", transparent=True)
        plt.close()
    except Exception as ex:
        print(f"No histogram made for {col}")
        exhandler(ex, module=os.path.basename(__file__))


# Pull out and plot possibly multimodal values

multimodal = []

# using standard alpha of 0.05 - change to whatever you like
alpha = 0.05

for x in list(df):
    try:
        p = modality.hartigan_diptest(df[x].values)

        if p > alpha:
            # uncomment out these print statements for more verbose output
            #       print(f"{x} appears to not be multimodal")
            #       print(f"p={p} \n")
            print("")

        else:
            multimodal.append(x)
    #       p < 0.05 suggests possible multimodality
    #       print(f"{x} appears to be multimodal")
    #       print(f"p={p} \n")

    except Exception as ex:
        exhandler(ex, module=os.path.basename(__file__))
#     print(f"Could not determine modality for {x} \n")
#     print("")


if not os.path.exists("multimodal"):
    os.makedirs("multimodal")


# make a plot for each possibly multimodal feature
for feature in multimodal:
    try:
        #     print(feature, "\n")
        df[feature].dropna().plot.kde(figsize=[12, 8])
        plt.legend([feature])
        plt.tight_layout()
        plt.savefig(f"multimodal/multimodal_{feature}.pdf", transparent=True)
        #     plt.show()
        plt.close()
    except Exception as ex:
        exhandler(ex, module=os.path.basename(__file__))
#     print(f"Could not make modality plot for {feature} \n")

try:
    df[multimodal].dropna().plot.kde(figsize=[12, 8])
    plt.legend(multimodal)
    plt.tight_layout()
    plt.savefig(f"multimodal/multimodal.pdf", transparent=True)
    #     plt.show()
    plt.close()
except Exception as ex:
    exhandler(ex, module=os.path.basename(__file__))


# Make empty lists to hold the outlier columns

near_outliers = []
far_outliers = []

# define thresholds - standards are 1.5 for near and 3 for far
near_thresh = 1.5
far_thresh = 3


if not os.path.exists("outliers"):
    os.makedirs("outliers")
if not os.path.exists("outliers/near_outliers"):
    os.makedirs("outliers/near_outliers")
if not os.path.exists("outliers/far_outliers"):
    os.makedirs("outliers/far_outliers")

# loop over the columns, pull out the non-missing values, test for outliers,
# and add the column to the list if it contains any outliers
for x in list(df):
    try:
        #     print(x, "\n")
        vals = df[x].values[~np.isnan(df[x].values)]
        try:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1

            # near outliers
            low_bound_near = q1 - (iqr * near_thresh)
            high_bound_near = q3 + (iqr * near_thresh)
            near_outlier_indices = np.where(
                (vals > high_bound_near) | (vals < low_bound_near)
            )
            if np.size(near_outlier_indices) >= 1:
                near_outliers.append(x)

            # far outliers
            low_bound_far = q1 - (iqr * far_thresh)
            high_bound_far = q3 + (iqr * far_thresh)
            far_outlier_indices = np.where(
                (vals > high_bound_far) | (vals < low_bound_far)
            )
            if np.size(far_outlier_indices) >= 1:
                far_outliers.append(x)

        except Exception as ex:
            print(f"Outlier detection failed for {x}")
            print("")
            exhandler(ex, module=os.path.basename(__file__))
    except Exception as ex:
        exhandler(ex, module=os.path.basename(__file__))


# Make individual plots...
try:
    for outlier in near_outliers:
        df[[outlier]].boxplot(whis=3)
        plt.tight_layout()
        plt.savefig(
            f"outliers/near_outliers/near_outlier_{outlier}.pdf", transparent=True
        )
        #     plt.show()
        plt.close()
except Exception as ex:
    exhandler(ex, module=os.path.basename(__file__))

try:
    for outlier in far_outliers:
        df[[outlier]].boxplot(whis=3)
        plt.tight_layout()
        plt.savefig(
            f"outliers/far_outliers/far_outlier_{outlier}.pdf", transparent=True
        )
        #     plt.show()
        plt.close()
except Exception as ex:
    exhandler(ex, module=os.path.basename(__file__))


# and combination plots. Notice the `plt.xticks(rotation=xx)`,
# which makes it so you can read the box plot much more easily.
try:
    df[far_outliers].boxplot(whis=3)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"outliers/far_outliers.pdf", transparent=True)
    #     plt.show()
    plt.close()
except Exception as ex:
    exhandler(ex, module=os.path.basename(__file__))

try:
    df[near_outliers].boxplot(whis=3)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"outliers/near_outliers.pdf", transparent=True)
    #     plt.show()
    plt.close()
except Exception as ex:
    exhandler(ex, module=os.path.basename(__file__))


print("#" * 80)
cowsay.cow("fin")
print("#" * 80)

