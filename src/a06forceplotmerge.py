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
import shutil
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
y = pd.read_hdf(config.RAW_DATA_FILE_H5, key="y")

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
        dest = config.FIGURES_DIR / "shap_images" / "force_plots"/ "processed" /f"{pt_num}_dx_{true_dx}"
        if not os.path.exists(dest):
            os.makedirs(dest) 
        for file_name in image_list:
            # full_file_name = os.path.join(src, file_name)
            if os.path.isfile(file_name):
                shutil.copy(file_name, dest)                                                               
    
        # load first image to get dimensions
        hts = []
        wds = []
        for i, image in enumerate(image_list):
            file_ = PdfFileReader(open(image_list[i], "rb"))
            file_sz = file_.getPage(0).mediaBox
            file_ht = file_sz.getHeight()
            file_wd = file_sz.getWidth()
            # print(file_ht, file_wd)
            hts.append(file_ht)
            wds.append(file_wd)
            
        # print(hts, wds)
        file0_ht = max(hts)
        file0_wd = max(wds)
        # print(file0_ht, file0_wd)

        tx = decimal.Decimal(float(file0_wd) * (1 / 100))
        ht_corr = 10
        # print(tx)

        # create blank pdf of the right size
        blank = PdfFileWriter()
        blank.addBlankPage(file0_wd+50, file0_ht * len(classes) + 100)
        blank_size = blank.getPage(0).mediaBox
        # print(blank_size)
        blank_size_ht = blank_size.getHeight()
        blank_size_wd = blank_size.getWidth()
        word_ty = float(blank_size_ht) - (40)
        # word_ty = float(word_ty)
        # print(word_ty)
        trans_ht = blank_size_ht // len(classes)
        blankpdf = "blank.pdf"
        with open(blankpdf, "wb") as outputStream:
            blank.write(outputStream)

        packet = io.BytesIO()
        # Create a new PDF with Reportlab
        can = canvas.Canvas(packet, pagesize=(blank_size_wd,blank_size_ht))
        can.setFont("Helvetica-Bold", 24)
        can.drawCentredString(float(file0_wd / 2), word_ty, f"True diagnosis: {true_dx.upper()}")
        can.showPage()
        can.save()

        # Move to the beginning of the StringIO buffer
        packet.seek(0)
        new_pdf = PdfFileReader(packet)

        # load blank pdf and the rest of the images
        file00 = PdfFileReader(open(blankpdf, "rb"))
        page = file00.getPage(0)
        page.mergePage(new_pdf.getPage(0))
        
        for i, image in enumerate(image_list):
            iter_file = PdfFileReader(open(image_list[i], "rb"))
            page.mergeTranslatedPage(iter_file.getPage(0), ty=trans_ht * i - ht_corr, tx=tx)
        
        output = PdfFileWriter()
        output.addPage(page)

        dest = config.FIGURES_DIR / "shap_images" / "force_plots"/ "processed"
        if pt_num in pt_nums_wrong:
            pdf_title = dest / f"forceplot_0_wrong_pt_{pt_num}_merged.pdf"
        else:
            pdf_title = dest / f"forceplot_0_right_pt_{pt_num}_merged.pdf"
        with open(
            pdf_title, "wb"
        ) as outputStream:
            output.write(outputStream)
    except Exception as ex:
            exhandler(ex, module=os.path.basename(__file__))


# os.remove(blankpdf)

timestr = time.strftime("_%Y-%m-%d-%H%M_")


print("finis")
