"""Smoke tests for the Streamlit app.

``AppTest`` runs app.py headless in-process, so CI catches a dashboard that
crashes on load — the kind of break that unit tests on the pipeline never see.
"""

import pytest
from streamlit.testing.v1 import AppTest

from churn.config import MODEL_PATH, PROJECT_ROOT

APP = str(PROJECT_ROOT / "app.py")
TIMEOUT = 180


@pytest.fixture(scope="module")
def app():
    if not MODEL_PATH.exists():
        pytest.skip("no trained model; run scripts/train.py --sample")
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    return at


def test_app_runs_without_exception(app):
    assert not app.exception


def test_the_three_tabs_are_present(app):
    assert len(app.tabs) == 3


def test_overview_reports_headline_metrics(app):
    labels = [m.label for m in app.metric]
    for expected in ("Events", "Users", "Churn rate"):
        assert expected in labels


def test_churn_rate_metric_is_realistic(app):
    value = next(m.value for m in app.metric if m.label == "Churn rate")
    assert 15.0 <= float(value.rstrip("%")) <= 30.0


def test_sidebar_exposes_the_cutoff_control(app):
    assert app.sidebar.date_input, "the cutoff control is the app's leakage story"


def test_charts_are_rendered(app):
    assert len(app.get("vega_lite_chart")) >= 3


def test_moving_the_cutoff_does_not_break_the_app():
    """The cutoff drives a full feature rebuild — the most fragile interaction."""
    if not MODEL_PATH.exists():
        pytest.skip("no trained model")
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    import datetime

    at.sidebar.date_input[0].set_value(datetime.date(2018, 11, 5)).run()
    assert not at.exception
