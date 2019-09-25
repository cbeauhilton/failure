import os
import pickle
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from cbh import (
    config,
)  # a convenience file I maintain to define all the filepaths and other things


def shap_reload(filename):
    f = h5py.File(filename, "r", libver="latest")

    shap_values_object = f["shap_values"]
    shap_values = shap_values_object[()]

    shap_expected_list = list(f.attrs["shap_expected"])
    classes = list(f.attrs["classes"])
    feature_names_short = list(f.attrs["feature_names_short"])
    feature_names = list(f.attrs["feature_names"])
    columns_in_order = list(f.attrs["columns_in_order"])

    return (
        shap_expected_list,
        shap_values,
        classes,
        feature_names_short,
        feature_names,
        columns_in_order,
    )


def model_reload(filename):
    loaded_model = pickle.load(open(filename, "rb"))
    return loaded_model


def make_pt_shap_plots(
    new_sample, classes, explainer, shap_values, feature_names, figure_directory
):
    timestr = time.strftime(
        "_%Y-%m-%d-%H%M_"
    )  # or some other way to keep all the images for a given prediction together
    for i, classname in enumerate(classes):
        shap.force_plot(
            explainer.expected_value[i],
            shap_values[i][0, :],
            new_sample,
            link="logit",
            matplotlib=True,
            feature_names=feature_names_short,
            show=False,
        )

        plt.title(
            f"{classname.upper()}", fontsize="xx-large", backgroundcolor="white", y=1.08
        )

        fig_dir = figure_directory / f"shap_images/deploy/force_plots"
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
            print(f"Made directory: {fig_dir}")

        plt.savefig(
            fig_dir / f"forceplot_{timestr}_{classname}_deploy.pdf",
            bbox_inches="tight",
            transparent=True,
        )

        plt.close()
        print(f"Generated figure for the prediction of {classname.upper()}.")


filename_shap = config.DEPLOY_MODEL_SHAP_H5
filename_model = config.DEPLOY_MODEL_PICKLE

shap_expected_list, shap_values, classes, feature_names_short, feature_names, columns_in_order = shap_reload(
    filename=filename_shap
)

model = model_reload(filename_model)
explainer = shap.TreeExplainer(model)

# Generate dictionary for sample on which we want a prediction
# should all be in the same order as the dataset
# (can check with 'columns_in_order' if you'd like)
new_sample_dict = {key: np.nan for key in feature_names}

# This is just for testing. Will fill the dict with user input in the app.
new_sample_dict["Absolute basophil count"] = 4.4
new_sample_dict["Absolute eosinophil count"] = 3.3
new_sample_dict["Absolute lymphocyte count"] = 2.2
new_sample_dict["Absolute monocyte count"] = 1.1
new_sample_dict["Absolute neutrophil count"] = 3.2
new_sample_dict["Hemoglobin"] = 7.3
new_sample_dict["Platelet count"] = 150
new_sample_dict["White blood cell count"] = 10.0
new_sample_dict["Age"] = 76
new_sample_dict["Female"] = 1
new_sample_dict["APC"] = 1
new_sample_dict["ASXL1"] = 1
new_sample_dict["BCOR"] = 1
new_sample_dict["BCORL1"] = 1
new_sample_dict["CUX1"] = 1
new_sample_dict["DDX54"] = 1
new_sample_dict["DHX29"] = 1
new_sample_dict["DNMT3A"] = 1
new_sample_dict["EED"] = 1
new_sample_dict["ERBB4"] = 1
new_sample_dict["ETV6"] = 1
new_sample_dict["EZH2"] = 1
new_sample_dict["FLT3"] = 1
new_sample_dict["GATA2"] = 1
new_sample_dict["GLI1"] = 1
new_sample_dict["GNB1"] = 1
new_sample_dict["GPR98"] = 1
new_sample_dict["JAK2"] = 1
new_sample_dict["KRA"] = 1
new_sample_dict["NF1"] = 1
new_sample_dict["NRA"] = 1
new_sample_dict["PHF6"] = 1
new_sample_dict["RUNX1"] = 1
new_sample_dict["SF3B1"] = 1
new_sample_dict["SRSF2"] = 1
new_sample_dict["STAG2"] = 1
new_sample_dict["TET2"] = 1
new_sample_dict["U2AF1"] = 1

# Grab all the genes, exploiting the upper case keys in the dict
genes = [k for k in new_sample_dict.keys() if k.isupper()]

# Add up all the mutations
genes_sum = sum(new_sample_dict[gene] for gene in genes)
new_sample_dict["Number of mutations"] = genes_sum

# The model and SHAP expect a row in a dataframe
new_sample = pd.DataFrame(new_sample_dict, index=[0])

model.predict_proba(new_sample)
shap_values = explainer.shap_values(new_sample)


make_pt_shap_plots(
    new_sample=new_sample,
    classes=classes,
    explainer=explainer,
    shap_values=shap_values,
    feature_names=feature_names_short,
    figure_directory=config.FIGURES_DIR,
)

# Then would load all predictions to the webapp.

print("")
