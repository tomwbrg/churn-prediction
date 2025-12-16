"""
Configuration constants for churn prediction pipeline.
"""

import pandas as pd
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
TRAIN_PATH = PROJECT_ROOT / "train.parquet"
TEST_PATH = PROJECT_ROOT / "test.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# =============================================================================
# TEMPORAL CUTOFFS
# =============================================================================

TRAIN_CUTOFF = pd.Timestamp("2018-11-10")
FINAL_CUTOFF = pd.Timestamp("2018-11-20")

# =============================================================================
# ANTI-LEAKAGE CONFIGURATION
# =============================================================================

LEAKY_PAGES = ["Cancellation Confirmation"]
CHURN_INDICATOR_PAGE = "Cancellation Confirmation"

# =============================================================================
# RANDOM SEED & CV SETTINGS
# =============================================================================

SEED = 42
CV_FOLDS = 10

# =============================================================================
# LIGHTGBM HYPERPARAMETERS
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
}

# =============================================================================
# SUBMISSION SETTINGS
# =============================================================================

THRESHOLD_PERCENTILE = 50.0

