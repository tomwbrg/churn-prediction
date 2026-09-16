"""Configuration constants for the churn prediction pipeline.

Paths can be overridden with environment variables so the same code runs
locally, in CI (on the committed sample) and inside Docker.
"""

import os
from pathlib import Path

import pandas as pd

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CHURN_DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = Path(os.environ.get("CHURN_MODEL_DIR", PROJECT_ROOT / "models"))

# The full dataset (568 MB) is never committed: see scripts/download_data.py.
# The sample is committed so that tests, CI and the Docker image are self-contained.
TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
SAMPLE_PATH = DATA_DIR / "train_sample.parquet"

MODEL_PATH = MODEL_DIR / "lgbm_churn.joblib"

# =============================================================================
# TEMPORAL CUTOFFS
# =============================================================================
# Features are built from events at or before the cutoff; the label looks at
# the 10 days that follow. Keeping these explicit is what prevents leakage.

TRAIN_CUTOFF = pd.Timestamp("2018-11-10")
FINAL_CUTOFF = pd.Timestamp("2018-11-20")

# =============================================================================
# ANTI-LEAKAGE CONFIGURATION
# =============================================================================

CHURN_PAGE = "Cancellation Confirmation"
LEAKY_PAGES = [CHURN_PAGE]

# Columns every raw event file must provide.
REQUIRED_COLUMNS = [
    "userId",
    "page",
    "time",
    "registration",
    "sessionId",
    "itemInSession",
    "level",
    "gender",
    "song",
    "artist",
    "length",
]

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

SEED = 42
CV_FOLDS = 10

# =============================================================================
# MODEL
# =============================================================================

LGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.02,
    "max_depth": 7,
    "num_leaves": 40,
    "subsample": 0.65,
    "colsample_bytree": 0.65,
    "min_child_samples": 40,
    "reg_alpha": 2.0,
    "reg_lambda": 2.0,
    "class_weight": "balanced",
    "verbose": -1,
    "random_state": SEED,
}

THRESHOLD_PERCENTILE = 50.0
