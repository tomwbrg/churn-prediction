# Music Streaming Churn Prediction

[![CI](https://github.com/tomwbrg/churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/tomwbrg/churn-prediction/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue.svg)](https://www.python.org/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.960-success.svg)](#results)
[![Docker Hub](https://img.shields.io/badge/docker%20hub-tomwbrg31%2Fchurn--prediction-2496ed.svg)](https://hub.docker.com/r/tomwbrg31/churn-prediction)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Predicting user churn for a music streaming platform from raw behavioural event
logs — 17.5 M events, 19 140 users — served as an interactive Streamlit
dashboard and shipped as a container.

Originally a Kaggle competition entry (**5th place**), refactored from a single
notebook into a tested, containerised, CI-backed project.

---

## Results

**Out-of-fold ROC-AUC: 0.9600** — LightGBM, 10-fold stratified cross-validation.

| | |
|---|---|
| OOF ROC-AUC | **0.9600** |
| Fold range | 0.9499 – 0.9682 |
| Spread across folds | 0.0183 |
| Users | 18 880 |
| Churn rate | 22.6 % |
| Features | 86 |
| Training time | 147 s |

The narrow spread across folds matters more than the headline number: it means
the score is a property of the model, not of one lucky split.

**Top predictors:** cancellation-page visits, days since last activity,
downgrade attempts, error rate, activity decline.

---

## Quick start

### Run with Docker

The published image carries the trained model and the sample data — nothing
else to download:

```bash
docker run -p 8501:8501 tomwbrg31/churn-prediction:1.0.0
```

Or build it yourself:

```bash
docker build -t churn-prediction .
docker run -p 8501:8501 churn-prediction
```

Open <http://localhost:8501>.

Image: [`tomwbrg31/churn-prediction`](https://hub.docker.com/r/tomwbrg31/churn-prediction)
(289 MB compressed, tags `1.0.0` and `latest`).

### Run locally

```bash
git clone https://github.com/tomwbrg/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
PYTHONPATH=src streamlit run app.py
```

On macOS, LightGBM needs the OpenMP runtime: `brew install libomp`.

---

## The app

Three tabs, all running on the committed sample so a fresh clone works
immediately:

- **Overview** — volumetry, churn rate, daily activity of churned vs retained
  users, page-event distribution, and per-feature distributions split by outcome.
- **Predict** — pick a user, get their churn probability and the features
  driving it, with their ground-truth label for comparison.
- **Model** — cross-validation metrics, per-fold AUC, feature importance.

The sidebar exposes the **feature cutoff**. Moving it rebuilds every feature
from scratch: it is the line that separates what the model may see from the
window the label is drawn from.

---

## Project structure

```
src/churn/
  config.py      paths (env-overridable), cutoffs, seed, LightGBM params
  data.py        loading, schema validation, temporal and leakage filters
  features.py    build_labels, build_features (86 features per user)
  model.py       prepare_xy, align_columns, train_cv, predict, persistence
tests/
  test_data.py       import & filtering functions
  test_features.py   labels, feature engineering, leakage guarantees
  test_app.py        the dashboard renders and survives interaction
scripts/
  train.py           train and persist the model
  download_data.py   fetch the full dataset
app.py           Streamlit dashboard
notebooks/       the original exploratory notebook, kept as-is
```

---

## Methodology

**The problem.** A user churns if they trigger a `Cancellation Confirmation`
event within the 10 days after a cutoff date. Given their event history up to
that cutoff, predict whether they will.

**Temporal construction.** Features are built only from events at or before
`TRAIN_CUTOFF` (2018-11-10); labels are read from the full window. Every
aggregate flows through `filter_before_cutoff`, so no feature can see the
period it is predicting.

**Leakage control.** `Cancellation Confirmation` *is* the label, so it is
excluded from all page-count features. `Cancel` — a visit to the cancellation
page without confirming — is deliberately kept: it is a legitimate and strong
predictor. The distinction is enforced by tests, not by convention.

**Features (86).** Activity and session aggregates, rate features, per-page
counts, engagement (thumbs, playlists, friends), diversity, recency windows
(3/7/14 days), activity trend and acceleration, subscription changes,
frustration indicators (errors, help visits), temporal patterns (weekend, peak
hour), and interaction terms.

**Model.** LightGBM with `class_weight='balanced'`, 10-fold stratified CV,
fold `i` seeded at `SEED + i`.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

55 tests, no network access and no external data required.

The suite concentrates on the data importing and filtering functions, because
that is where a silent bug corrupts every downstream result without ever
raising. The leakage tests are the load-bearing ones — in particular
`test_events_after_the_cutoff_do_not_change_any_feature`, which replays the
same history with extra future activity and asserts the features come out
byte-identical. A model that sees the future scores beautifully in validation
and is worthless in production, and no metric reveals it.

These tests earned their keep immediately: they caught `page_counts.get(page, 0)`
returning a bare integer instead of a Series whenever a page was absent from a
slice of the data, which silently broke every expression built on it.

---

## Reproducibility

| Concern | How it is handled |
|---|---|
| **Data** | The full dataset (568 MB) is too large for Git. A **stratified 800-user sample** is committed instead — sampled by *user* with `seed=42`, preserving the 22.3 % churn rate and ~911 events/user of the full set. Tests, CI, Docker and the app all run on it. |
| **Randomness** | A single `SEED = 42` in `config.py`; fold `i` uses `SEED + i`. |
| **Dependencies** | Pinned to exact versions in `requirements.txt`. |
| **Environment** | The Docker image pins the base image, the OS packages and every Python dependency. |
| **Paths** | Overridable via `CHURN_DATA_DIR` / `CHURN_MODEL_DIR`, so the same code runs locally, in CI and in the container. |
| **Model** | The trained model is committed with the feature order it was trained on, so the app can never feed columns in the wrong order. |
| **Python versions** | CI runs the suite on 3.11 and 3.12 — the floor scikit-learn 1.8 imposes. |

Retraining from scratch, once the full dataset is in `data/`:

```bash
python scripts/download_data.py --check   # verify what is present
PYTHONPATH=src python scripts/train.py    # ~150 s, writes models/
```

Or on the sample alone, in a few seconds:

```bash
PYTHONPATH=src python scripts/train.py --sample
```

---

## CI

Every push runs three jobs:

1. **Lint** — `ruff format --check` and `ruff check`.
2. **Tests** — the full suite on Python 3.11 and 3.12, with coverage.
3. **Docker** — builds the image, starts the container, and polls
   `/_stcore/health` until the app answers. A build that succeeds but produces
   an app that dies on startup fails the pipeline.

---

## Dataset

Event logs from a music streaming service: 17.5 M rows, 19 140 users, one row
per user action (song played, thumbs up, page visit, subscription change) over
2018-10-01 → 2018-11-19.

The committed sample has direct identity columns (`firstName`, `lastName`)
removed — they carry no signal for the model and do not belong in a public
repository.

---

## License

MIT — see [LICENSE](LICENSE).
