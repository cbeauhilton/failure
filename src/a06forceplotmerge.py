import csv
import decimal
import io
import os
import random
import sys
import time
import traceback
import warnings
from glob import glob
from itertools import cycle
from pathlib import Path

import h5py
import jsonpickle
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from PIL import Image
from PyPDF2 import PdfFileReader, PdfFileWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from scipy import interp
from sklearn import datasets, svm
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.multiclass import type_of_target

from cbh import config
from cbh.dumb import get_xkcd_colors
from cbh.exhandler import exhandler

colorz = get_xkcd_colors()
colors = cycle(colorz["hex"])

warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
print("Loading", os.path.basename(__file__))


# Load data
data = pd.read_hdf(config.RAW_DATA_FILE_H5, key="data")
data = data.round(2)
y = data["diagnosis"]
classes = np.unique(y)  # retrieve all class names
n_force_plots = config.N_FORCE_PLOTS
pdfheight = 8.5 * 72
pdfwidth = 11 * 72


force_plot_csv = pd.read_csv(config.TABLES_DIR / "pt_nums.csv")
pt_nums_wrong = force_plot_csv["incorrectly_classified"].values
pt_nums_right = force_plot_csv["correctly_classified"].values
pt_nums = list(pt_nums_wrong) + list(pt_nums_right)

print(pt_nums)

os.chdir(config.FIGURES_DIR / "shap_images" / "force_plots")

print(Path.cwd())

for pt_num in pt_nums:
    try:
        print("\nPt_num:", pt_num)
        true_dx = data.at[pt_num, "diagnosis"]
        print("True dx:", true_dx)
        # grab files
        files = Path.cwd().glob(f"*_pt_{pt_num}.pdf")
        # print("Files: \n", files)
        # empty image list
        image_list = []
        # fill image list
        for file in files:
            file = Path.absolute(file)
            file = file.as_posix()
            image_list.append(file)
        # check image list
        print("Image List:")
        for i in range(len(classes)):
            print(image_list[i])

        # load first image to get dimensions
        file0 = PdfFileReader(open(image_list[0], "rb"))
        file0_sz = file0.getPage(0).mediaBox
        file0_ht = file0_sz.getHeight()
        file0_wd = file0_sz.getWidth()
        # print(file0_ht, file0_wd)
        tx = decimal.Decimal(float(file0_wd) * (1 / 15))
        ht_corr = 50
        # print(tx)

        # create blank pdf of the right size
        blank = PdfFileWriter()
        blank.addBlankPage(file0_ht * len(classes), file0_wd)
        blank_size = blank.getPage(0).mediaBox
        blank_size_ht = blank_size.getHeight()
        word_ty = float(blank_size_ht) - (40)
        # word_ty = float(word_ty)
        # print(word_ty)
        trans_ht = blank_size_ht // len(classes)
        blankpdf = "blank.pdf"
        with open(blankpdf, "wb") as outputStream:
            blank.write(outputStream)

        packet = io.BytesIO()
        # Create a new PDF with Reportlab
        can = canvas.Canvas(packet, pagesize=letter)
        can.setFont("Helvetica-Bold", 24)
        can.drawString(file0_wd / 2, word_ty, f"{true_dx.upper()}")
        can.showPage()
        can.save()

        # Move to the beginning of the StringIO buffer
        packet.seek(0)
        new_pdf = PdfFileReader(packet)

        # load blank pdf and the rest of the images
        file00 = PdfFileReader(open(blankpdf, "rb"))
        file1 = PdfFileReader(open(image_list[1], "rb"))
        file2 = PdfFileReader(open(image_list[2], "rb"))
        file3 = PdfFileReader(open(image_list[3], "rb"))
        file4 = PdfFileReader(open(image_list[4], "rb"))
        output = PdfFileWriter()

        # merge
        page = file00.getPage(0)
        page.mergePage(new_pdf.getPage(0))
        page.mergeTranslatedPage(file0.getPage(0), ty=-ht_corr, tx=tx)
        page.mergeTranslatedPage(file1.getPage(0), ty=trans_ht - ht_corr, tx=tx)
        page.mergeTranslatedPage(file2.getPage(0), ty=trans_ht * 2 - ht_corr, tx=tx)
        page.mergeTranslatedPage(file3.getPage(0), ty=trans_ht * 3 - ht_corr, tx=tx)
        page.mergeTranslatedPage(file4.getPage(0), ty=trans_ht * 4 - ht_corr, tx=tx)
        output.addPage(page)
        if pt_num in pt_nums_wrong:
            pdf_title = f"forceplot_0_wrong_pt_{pt_num}_merged.pdf"
        else:
            pdf_title = f"forceplot_0_right_pt_{pt_num}_merged.pdf"
        with open(
            pdf_title, "wb"
        ) as outputStream:
            output.write(outputStream)
    except Exception as ex:
            exhandler(ex)


# os.remove(blankpdf)

timestr = time.strftime("_%Y-%m-%d-%H%M_")


print("finis")
