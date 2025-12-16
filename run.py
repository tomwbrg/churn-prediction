#!/usr/bin/env python3
"""
Churn Prediction Pipeline - Main Entrypoint

Usage:
    python run.py              # Full pipeline (train + predict)
    python run.py --eda        # EDA only
    python run.py --train      # Train only (saves OOF + importance)
    python run.py --predict    # Predict only (requires prior training)
"""

import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    TRAIN_PATH, TEST_PATH, OUTPUT_DIR,
    TRAIN_CUTOFF, FINAL_CUTOFF,
    LEAKY_PAGES, SEED, CV_FOLDS,
    LGB_PARAMS, THRESHOLD_PERCENTILE,
)
from features import build_features, build_labels
from model import train_cv, predict


# =============================================================================
# EDA FUNCTION
# =============================================================================

def run_eda(events_df: pd.DataFrame, y: pd.Series = None, cutoff: pd.Timestamp = None):
    """
    Run minimal, leakage-safe EDA.
    
    Args:
        events_df: Raw events DataFrame
        y: Optional labels Series
        cutoff: Cutoff for behavior-based EDA (default: TRAIN_CUTOFF)
    """
    if cutoff is None:
        cutoff = TRAIN_CUTOFF
    
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Dataset shape & users
    print(f"\n Dataset Shape")
    print(f"   Total events: {len(events_df):,}")
    print(f"   Unique users: {events_df['userId'].nunique():,}")
    print(f"   Columns: {list(events_df.columns)}")
    
    # Timestamp range
    print(f"\n Timestamp Range")
    print(f"   Min: {events_df['time'].min()}")
    print(f"   Max: {events_df['time'].max()}")
    print(f"   Span: {(events_df['time'].max() - events_df['time'].min()).days} days")
    
    # Churn rate (if labels provided)
    if y is not None:
        churn_rate = y.mean() * 100
        print(f"\n  Churn Rate")
        print(f"   Churned: {y.sum():,} ({churn_rate:.2f}%)")
        print(f"   Retained: {len(y) - y.sum():,} ({100 - churn_rate:.2f}%)")
    
    # Top pages BEFORE cutoff (leakage-safe)
    df_safe = events_df[events_df["time"] <= cutoff]
    print(f"\n Top Pages (before cutoff {cutoff.date()})")
    page_counts = df_safe["page"].value_counts().head(15)
    for page, count in page_counts.items():
        leaky = " LEAKY" if page in LEAKY_PAGES else ""
        print(f"   {page:<35} {count:>10,}{leaky}")
    
    # Events per user summary
    print(f"\n Events per User (before cutoff)")
    events_per_user = df_safe.groupby("userId").size()
    print(f"   Mean:   {events_per_user.mean():.1f}")
    print(f"   Median: {events_per_user.median():.1f}")
    print(f"   Min:    {events_per_user.min()}")
    print(f"   Max:    {events_per_user.max()}")
    
    # Missingness summary
    print(f"\n Missingness Summary")
    missing = events_df.isna().sum()
    for col in events_df.columns:
        if missing[col] > 0:
            pct = missing[col] / len(events_df) * 100
            print(f"   {col:<20} {missing[col]:>10,} ({pct:.2f}%)")
    if missing.sum() == 0:
        print("    No missing values!")
    
    print("\n" + "=" * 60)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    """Load train and test data."""
    print(f"\n Loading data...")
    print(f"   Train: {TRAIN_PATH}")
    print(f"   Test: {TEST_PATH}")
    
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    print(f"   Train shape: {train_df.shape}")
    print(f"   Test shape: {test_df.shape}")
    
    return train_df, test_df


def run_training(train_df: pd.DataFrame) -> tuple:
    """Run training pipeline."""
    print(f"\n{'='*60}")
    print("TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f" Train cutoff: {TRAIN_CUTOFF}")
    
    # Build features
    print(f"\n Building features...")
    train_features = build_features(train_df, TRAIN_CUTOFF)
    print(f"   Features shape: {train_features.shape}")
    
    # Build labels
    labels = build_labels(train_df)
    train_features = train_features[train_features["userId"].isin(labels.index)]
    y = labels.loc[train_features["userId"]].values
    
    print(f"   Labels: {sum(y)} churned / {len(y) - sum(y)} retained ({sum(y)/len(y)*100:.2f}% churn)")
    
    # Prepare X
    X = train_features.drop(columns=["userId"])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    user_ids = train_features["userId"].values
    
    print(f"   X shape: {X.shape}")
    
    # Train
    oof_preds, models, fi_df = train_cv(X, y, LGB_PARAMS, CV_FOLDS, SEED)
    
    # Save artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    oof_df = pd.DataFrame({
        "userId": user_ids,
        "true_label": y,
        "oof_prediction": oof_preds
    })
    oof_path = OUTPUT_DIR / "oof_preds.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"\n Saved: {oof_path}")
    
    fi_path = OUTPUT_DIR / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    print(f" Saved: {fi_path}")
    
    # Print top features
    print(f"\n Top 15 Features:")
    for i, row in fi_df.head(15).iterrows():
        print(f"   {i+1:2d}. {row['feature']:<35} {row['importance_mean']:.1f}")
    
    return models, X.columns.tolist()


def run_prediction(test_df: pd.DataFrame, models: list, feature_cols: list):
    """Run prediction pipeline."""
    print(f"\n{'='*60}")
    print("PREDICTION PIPELINE")
    print(f"{'='*60}")
    print(f"Test cutoff: {FINAL_CUTOFF}")
    
    # Build features
    print(f"\nBuilding test features...")
    test_features = build_features(test_df, FINAL_CUTOFF)
    print(f"   Features shape: {test_features.shape}")
    
    # Align columns
    test_user_ids = test_features["userId"].values
    X_test = test_features.drop(columns=["userId"])
    
    # Use only common columns
    common_cols = [c for c in feature_cols if c in X_test.columns]
    missing_cols = [c for c in feature_cols if c not in X_test.columns]
    
    if missing_cols:
        print(f"    Missing columns in test (set to 0): {missing_cols}")
        for col in missing_cols:
            X_test[col] = 0
    
    X_test = X_test[feature_cols]
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"   X_test shape: {X_test.shape}")
    
    # Predict
    print(f"\nGenerating predictions...")
    preds = predict(models, X_test)
    
    print(f"   Predictions: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}")
    
    # Create submission
    threshold = np.percentile(preds, THRESHOLD_PERCENTILE)
    submission = pd.DataFrame({
        "id": test_user_ids,
        "target": (preds >= threshold).astype(int)
    })
    
    churn_rate = submission["target"].mean() * 100
    print(f"   Threshold (p{THRESHOLD_PERCENTILE}): {threshold:.4f}")
    print(f"   Predicted churn rate: {churn_rate:.2f}%")
    
    # Save submission
    sub_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\n Saved: {sub_path}")
    
    return submission


def main():
    parser = argparse.ArgumentParser(description="Churn Prediction Pipeline")
    parser.add_argument("--eda", action="store_true", help="Run EDA only")
    parser.add_argument("--train", action="store_true", help="Train only")
    parser.add_argument("--predict", action="store_true", help="Predict only (requires prior training)")
    args = parser.parse_args()
    
    set_seeds(SEED)
    
    print("\n" + "=" * 60)
    print("CHURN PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print(f"CV Folds: {CV_FOLDS}")
    print(f"Train cutoff: {TRAIN_CUTOFF}")
    print(f"Test cutoff: {FINAL_CUTOFF}")
    print(f"Leaky pages: {LEAKY_PAGES}")
    
    # Load data
    train_df, test_df = load_data()
    
    if args.eda:
        # EDA only
        labels = build_labels(train_df)
        run_eda(train_df, y=labels, cutoff=TRAIN_CUTOFF)
        return
    
    if args.predict:
        # Predict only - would need saved models (not implemented for simplicity)
        print(" --predict requires models from prior training.")
        print("   Run without flags for full pipeline.")
        return
    
    if args.train:
        # Train only
        run_training(train_df)
        return
    
    # Full pipeline (default)
    models, feature_cols = run_training(train_df)
    run_prediction(test_df, models, feature_cols)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"📁 Outputs saved to: {OUTPUT_DIR}")
    print("   - submission.csv")
    print("   - oof_preds.csv")
    print("   - feature_importance.csv")


if __name__ == "__main__":
    main()

