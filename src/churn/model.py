"""Model training, prediction and persistence."""

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from churn.config import CV_FOLDS, LGB_PARAMS, MODEL_PATH, SEED


def prepare_xy(
    features_df: pd.DataFrame, labels: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Split a feature frame into a clean X matrix and (optionally) y.

    Ratio features divide by counts that can be zero, so infinities are real:
    they are replaced with 0 rather than left to blow up the model.
    """
    X = features_df.drop(columns=["userId", "churned"], errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    if labels is None:
        return X, None
    y = features_df["userId"].map(labels).fillna(0).astype(int)
    return X, y


def align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both matrices to their shared columns, in a stable order.

    Page-count features are built from the pages actually present, so a page
    absent from one split silently shifts the column layout. Sorting keeps the
    feature order reproducible across runs.
    """
    common = sorted(X_train.columns.intersection(X_test.columns))
    return X_train[common], X_test[common]


def train_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    params: dict = None,
    n_splits: int = CV_FOLDS,
    seed: int = SEED,
    verbose: bool = True,
) -> dict:
    """Train LightGBM with stratified K-fold cross-validation.

    Args:
        X: Feature matrix.
        y: Binary target.
        X_test: Optional test matrix; test predictions are averaged over folds.
        params: LightGBM parameters (defaults to config.LGB_PARAMS).
        n_splits: Number of folds.
        seed: Base random seed; fold ``i`` uses ``seed + i``.
        verbose: Print per-fold AUC.

    Returns:
        Dict with oof_preds, test_preds, models, importance_df, fold_aucs
        and oof_auc.
    """
    params = dict(params or LGB_PARAMS)
    params.pop("random_state", None)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    models: List[LGBMClassifier] = []
    importances, fold_aucs = [], []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params, random_state=seed + fold)
        model.fit(X_tr, y_tr)

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        if X_test is not None:
            test_preds += model.predict_proba(X_test)[:, 1] / n_splits

        fold_auc = roc_auc_score(y_val, val_preds)
        fold_aucs.append(fold_auc)
        if verbose:
            print(f"Fold {fold + 1}/{n_splits} AUC: {fold_auc:.4f}")

        models.append(model)
        importances.append(model.feature_importances_)

    oof_auc = roc_auc_score(y, oof_preds)
    if verbose:
        print(f"Overall OOF AUC: {oof_auc:.4f}")

    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": np.mean(importances, axis=0)}
    ).sort_values("importance", ascending=False, ignore_index=True)

    return {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "models": models,
        "importance_df": importance_df,
        "fold_aucs": fold_aucs,
        "oof_auc": oof_auc,
    }


def predict(models: List[LGBMClassifier], X: pd.DataFrame) -> np.ndarray:
    """Average churn probabilities over every fold model."""
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def apply_threshold(preds: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    """Turn probabilities into binary labels via a percentile threshold.

    Returns the labels and the threshold value itself.
    """
    threshold = float(np.percentile(preds, percentile))
    return (preds >= threshold).astype(int), threshold


def save_model(bundle: dict, path: Path = MODEL_PATH) -> Path:
    """Persist models plus the feature order they were trained on."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model(path: Path = MODEL_PATH) -> dict:
    """Load a persisted model bundle."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/train.py` to train and save it."
        )
    return joblib.load(path)
