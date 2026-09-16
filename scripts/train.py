"""Train the churn model and persist it for the Streamlit app.

    python scripts/train.py --sample    # fast, on the committed 800-user sample
    python scripts/train.py             # full dataset (needs data/train.parquet)

The saved bundle carries the fold models *and* the feature order they were
trained on, so the app can never feed columns in the wrong order.
"""

import argparse
import json
import time

from churn.config import (
    CV_FOLDS,
    FINAL_CUTOFF,
    LGB_PARAMS,
    MODEL_DIR,
    SEED,
    TRAIN_CUTOFF,
)
from churn.data import load_data
from churn.features import build_features, build_labels
from churn.model import prepare_xy, save_model, train_cv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="train on the committed sample")
    parser.add_argument("--folds", type=int, default=CV_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    started = time.time()

    print("Loading events...")
    train_df, _ = load_data(sample=args.sample)
    print(f"  {len(train_df):,} events, {train_df['userId'].nunique():,} users")

    # Labels come from the full window; features stop at the cutoff.
    print(f"Building features (cutoff {TRAIN_CUTOFF.date()})...")
    labels = build_labels(train_df)
    features_df = build_features(train_df, TRAIN_CUTOFF)
    X, y = prepare_xy(features_df, labels)
    print(f"  {X.shape[1]} features, {len(X):,} users, churn {y.mean():.2%}")

    print(f"Training LightGBM, {args.folds}-fold stratified CV...")
    result = train_cv(X, y, params=LGB_PARAMS, n_splits=args.folds, seed=args.seed)

    bundle = {
        "models": result["models"],
        "feature_names": list(X.columns),
        "oof_auc": result["oof_auc"],
        "fold_aucs": result["fold_aucs"],
        "importance": result["importance_df"].to_dict("records"),
        "trained_on": "sample" if args.sample else "full",
        "n_users": int(len(X)),
        "churn_rate": float(y.mean()),
        "seed": args.seed,
        "folds": args.folds,
        "train_cutoff": str(TRAIN_CUTOFF),
        "final_cutoff": str(FINAL_CUTOFF),
    }
    path = save_model(bundle)

    # A plain-text summary the README and the app can quote without unpickling.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        k: bundle[k]
        for k in ("oof_auc", "fold_aucs", "trained_on", "n_users", "churn_rate", "seed", "folds")
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nOOF AUC {result['oof_auc']:.4f} | saved to {path}")
    print(f"Done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
