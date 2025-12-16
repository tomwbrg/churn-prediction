"""
Model training and prediction for churn prediction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier


def train_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    n_splits: int,
    seed: int
) -> Tuple[np.ndarray, List[LGBMClassifier], pd.DataFrame]:
    """
    Train LightGBM with stratified K-fold cross-validation.
    
    Args:
        X: Feature matrix
        y: Target labels
        params: LightGBM parameters
        n_splits: Number of CV folds
        seed: Random seed
    
    Returns:
        Tuple of (oof_predictions, list_of_models, feature_importance_df)
    """
    oof_preds = np.zeros(len(X))
    models = []
    importances = []
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    print(f"\n{'='*60}")
    print(f"Training {n_splits}-fold CV")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model = LGBMClassifier(**params, random_state=seed + fold)
        model.fit(X_tr, y_tr)
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        fold_auc = roc_auc_score(y_val, val_preds)
        print(f"  Fold {fold + 1}/{n_splits} - AUC: {fold_auc:.4f}")
        
        models.append(model)
        importances.append(pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
            "fold": fold
        }))
    
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n Overall OOF AUC: {overall_auc:.4f}")
    
    # Aggregate feature importance
    fi_df = pd.concat(importances)
    fi_agg = fi_df.groupby("feature")["importance"].agg(["mean", "std"]).reset_index()
    fi_agg.columns = ["feature", "importance_mean", "importance_std"]
    fi_agg = fi_agg.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    
    return oof_preds, models, fi_agg


def predict(models: List[LGBMClassifier], X_test: pd.DataFrame) -> np.ndarray:
    """
    Generate predictions by averaging across all CV models.
    
    Args:
        models: List of trained LightGBM models
        X_test: Test feature matrix
    
    Returns:
        Array of averaged probability predictions
    """
    preds = np.zeros(len(X_test))
    
    for model in models:
        preds += model.predict_proba(X_test)[:, 1] / len(models)
    
    return preds

