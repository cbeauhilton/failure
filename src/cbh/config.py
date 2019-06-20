from pathlib import Path
import os

# print("Loading", os.path.basename(__file__))

########################### directories ###########################
PROJECT_DIR = Path(r"C:/Users/hiltonc/Desktop/failure/")
RAW_DATA_DIR = PROJECT_DIR / "data/raw/"
INTERIM_DATA_DIR = PROJECT_DIR / "data/interim/"
EXTERNAL_DATA_DIR = PROJECT_DIR / "data/external/"
PROCESSED_DATA_DIR = PROJECT_DIR / "data/processed/"
MODELS_DIR = PROJECT_DIR / "models/"

NOTEBOOK_DIR = PROJECT_DIR / "notebooks/"

FIGURES_DIR = PROJECT_DIR / "reports/figures/"
METRIC_FIGS_DIR = FIGURES_DIR / "metrics/"

TABLES_DIR = PROJECT_DIR / "reports/tables/"

TEXT_DIR = PROJECT_DIR / "reports/text/pygen/"
TEX_SECTIONS_DIR = PROJECT_DIR / "reports/text/sections/"
TEX_TABLE_DIR = PROJECT_DIR / "reports/text/tables/"
DOCS_DIR = PROJECT_DIR / "docs/"
########################### files ###########################
RAW_AHED_MPN_FILE = RAW_DATA_DIR / "ahed_mpn.xlsx"
RAW_CCF_FILE = RAW_DATA_DIR / "ccf.csv"
RAW_MLL_CMML_FILE = RAW_DATA_DIR / "mll_cmml.xlsx"
RAW_MLL_MDS_FILE = RAW_DATA_DIR / "mll_mds.xlsx"
RAW_DATA_FILE_H5 = RAW_DATA_DIR / "data.h5"
RAW_H5_KEY = "raw"


# PROCESSED_FINAL = CLEAN_PHASE_10
# PROCESSED_FINAL_DESCRIPTIVE = CLEAN_PHASE_11_TABLEONE


# TRAINING_REPORTS = TABLES_DIR / "classifiertrainingreports.csv"
# REGRESSOR_TRAINING_REPORTS = TABLES_DIR / "regressortrainingreports.csv"
# PAPER_NUMBERS = TABLES_DIR / "papernumbers.csv"
# RESULTS_TEX = TEX_TABLE_DIR /  "all_results_df.tex"

# ########################### hyperparameters ###########################
# NUMERIC_IMPUTER = "median"
# SCALER = "StandardScaler"
# CATEGORICAL_IMPUTER_STRATEGY = "constant"
# CATEGORICAL_IMPUTER_FILL_VALUE = "missing"
# ONE_HOT_ENCODER = 1
SEED = 42


PARAMS_LGBM = {
    # "boosting_type": "gbdt",
    # "colsample_bytree": 0.707630032256903,
    "is_unbalance": "true",
    # "learning_rate": 0.010302298912236304,
    # "max_depth": -1,
    # "min_child_samples": 10,
    # "min_child_weight": 0.001,
    # "min_split_gain": 0.0,
    # "n_estimators": 568,
    # "n_jobs": -1,
    # "num_leaves": 99,
    # "num_rounds": 10_000_000,
    # "objective": "binary",
    "objective": "multiclass",
    # "objective": "multiclassova",
    "predict_contrib": True,
    "random_state": SEED,
    # "reg_alpha": 0.5926734167821595,
    # "reg_lambda": 0.1498749826768534,
    # "seed": SEED,
    # "silent": False,
    # "subsample_for_bin": 240000,
    # "subsample_freq": 0,
    # "subsample": 0.6027609913849075,
    # "boost_from_average": True,
    # "importance_type": "split",
    # "num_threads": 8,
    # "verbosity": -1,
    "verbose": -1,
}


GENE_COLS = [
    "apc",
    "asxl1",
    "bcor",
    "bcorl1",
    "c7orf55",
    "cbl",
    "ccdc42b",
    "cdh23",
    "cebpa",
    "cftr",
    "csf1r",
    "cux1",
    "ddx41",
    "ddx54",
    "dhx29",
    "dnmt3a",
    "eed",
    "erbb4",
    "etv6",
    "ezh2",
    "flt3",
    "gata2",
    "gli1",
    "gli2",
    "gnb1",
    "gpr98",
    "idh1",
    "idh2",
    "irf4",
    "jak2",
    "jak3",
    "kdm6a",
    "kit",
    "kras",
    "mecom",
    "med12",
    "mll",
    "nf1",
    "npm1",
    "nras",
    "ogt",
    "phf6",
    "prpf8",
    "ptch1",
    "ptpn11",
    "rad21",
    "rnf25",
    "runx1",
    "setbp1",
    "sf3b1",
    "simc1",
    "smc3",
    "srsf2",
    "stag2",
    "stat3",
    "suz12",
    "tet2",
    "tp53",
    "u2af1",
    "u2af2",
    "wt1",
    "zrsr2",
]

LAB_COLS = [
    "abs bas",
    "abs eos",
    "abs lym",
    "abs mono",
    "abs neut",
    "bas %",
    "eos %",
    "hgb",
    "lym %",
    "mch",
    "mchc",
    "mcv",
    "megakaryocytes",
    "mono %",
    "mpv",
    "neut %",
    "percent bm blasts",
    "plt",
    "rdw",
    "wbc",
]

PT_COLS = ["age", "diagnosis", "gender", "id", "splenomegaly"]
