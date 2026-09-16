"""Data loading and filtering.

These are the functions the test suite targets: they are where a silent bug
would quietly corrupt every downstream result, and where temporal leakage
would creep in.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from churn.config import (
    CHURN_PAGE,
    LEAKY_PAGES,
    REQUIRED_COLUMNS,
    SAMPLE_PATH,
    TEST_PATH,
    TRAIN_PATH,
)


class SchemaError(ValueError):
    """Raised when an event file does not carry the expected columns."""


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Check that a raw events frame has the columns the pipeline needs.

    Args:
        df: Raw events DataFrame.

    Returns:
        The same DataFrame, unchanged.

    Raises:
        SchemaError: If any required column is missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SchemaError(f"missing required column(s): {', '.join(sorted(missing))}")
    return df


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the two timestamp columns to datetime and userId to string.

    Parquet already stores the right types, but a CSV export or a hand-built
    test fixture may not. Normalizing here keeps every downstream comparison
    (``time <= cutoff``) well defined.
    """
    out = df.copy()
    for col in ("time", "registration"):
        if col in out.columns and not pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col])
    if "userId" in out.columns:
        out["userId"] = out["userId"].astype(str)
    return out


def load_events(path: Path, columns: Optional[list] = None) -> pd.DataFrame:
    """Read an event log from parquet, validating and normalizing it.

    Args:
        path: Path to the parquet file.
        columns: Optional subset of columns to read (saves memory on the
            568 MB training file).

    Returns:
        A validated, type-normalized events DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist, with a pointer to the
            download script rather than a bare traceback.
        SchemaError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` to fetch "
            f"the full dataset, or use the committed sample at {SAMPLE_PATH}."
        )
    df = pd.read_parquet(path, columns=columns)
    return normalize_types(validate_schema(df))


def load_data(sample: bool = False) -> tuple:
    """Load the train and test event logs.

    Args:
        sample: When True, load the small committed sample for both splits.
            This is what CI and the Docker image use.

    Returns:
        A ``(train_df, test_df)`` tuple.
    """
    if sample:
        df = load_events(SAMPLE_PATH)
        return df, df
    return load_events(TRAIN_PATH), load_events(TEST_PATH)


def filter_before_cutoff(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Keep only events at or before ``cutoff``.

    This single line is the pipeline's main defence against temporal leakage:
    every feature is computed on the output of this function, so no feature can
    see the future the model is asked to predict.
    """
    return df[df["time"] <= cutoff].copy()


def drop_leaky_pages(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the events that directly reveal the target.

    A ``Cancellation Confirmation`` row *is* the label. Counting it as a
    feature would give a model that scores perfectly and predicts nothing.
    """
    return df[~df["page"].isin(LEAKY_PAGES)].copy()


def filter_valid_users(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no usable userId.

    Logged-out sessions carry an empty userId; they cannot be attributed to a
    user and would otherwise be aggregated into a spurious extra user.
    """
    users = df["userId"].astype(str).str.strip()
    return df[users.ne("") & users.ne("nan") & df["userId"].notna()].copy()


def get_churned_users(df: pd.DataFrame) -> set:
    """Return the set of userIds that confirmed a cancellation."""
    return set(df.loc[df["page"] == CHURN_PAGE, "userId"].unique())
