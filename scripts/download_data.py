"""Fetch the full event dataset.

The repository ships a stratified 800-user sample (`data/train_sample.parquet`)
so that tests, CI, the Docker image and the app all work on a bare clone. The
full dataset (568 MB train + 135 MB test) is only needed to retrain the model
from scratch, and is too large to version.

    python scripts/download_data.py                 # from $CHURN_DATA_URL
    python scripts/download_data.py --check         # report what is present

Set CHURN_DATA_URL to the directory holding train.parquet and test.parquet, or
drop the two files into data/ by hand.
"""

import argparse
import os
import sys
import urllib.request

from churn.config import DATA_DIR, SAMPLE_PATH, TEST_PATH, TRAIN_PATH

FILES = {"train.parquet": TRAIN_PATH, "test.parquet": TEST_PATH}


def human(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def report() -> bool:
    """Print what is available locally. Returns True if the full set is there."""
    print(f"data directory: {DATA_DIR}")
    ok = True
    for name, path in FILES.items():
        if path.exists():
            print(f"  [ok]      {name:16} {human(path.stat().st_size)}")
        else:
            print(f"  [missing] {name:16}")
            ok = False
    if SAMPLE_PATH.exists():
        print(f"  [ok]      {SAMPLE_PATH.name:16} {human(SAMPLE_PATH.stat().st_size)} (committed)")
    return ok


def download(base_url: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in FILES.items():
        if path.exists():
            print(f"{name}: already present, skipping")
            continue
        url = f"{base_url.rstrip('/')}/{name}"
        print(f"{name}: downloading from {url}")
        urllib.request.urlretrieve(url, path)
        print(f"{name}: done ({human(path.stat().st_size)})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report and exit")
    parser.add_argument("--url", default=os.environ.get("CHURN_DATA_URL"))
    args = parser.parse_args()

    if args.check:
        return 0 if report() else 1

    if not args.url:
        print(
            "No source URL. Either set CHURN_DATA_URL (or pass --url) to the "
            "directory holding train.parquet and test.parquet, or copy the two "
            f"files into {DATA_DIR} by hand.\n\n"
            "Everything except retraining works without them, on the committed "
            f"sample: {SAMPLE_PATH.name}",
            file=sys.stderr,
        )
        return 1

    download(args.url)
    return 0 if report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
