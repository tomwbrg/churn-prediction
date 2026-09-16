"""Shared fixtures.

Most tests run on a hand-built event log rather than the sample file: a
fixture whose every row is known makes assertions exact instead of
approximate. The sample is used only where realistic data matters.
"""

import pandas as pd
import pytest

from churn.config import CHURN_PAGE, SAMPLE_PATH


def make_event(user_id, page, time, **overrides):
    """Build one raw event row with sane defaults for every required column."""
    event = {
        "userId": user_id if user_id is None else str(user_id),
        "page": page,
        "time": pd.Timestamp(time),
        "registration": pd.Timestamp("2018-09-01"),
        "sessionId": 1,
        "itemInSession": 0,
        "level": "paid",
        "gender": "F",
        "song": "Song" if page == "NextSong" else None,
        "artist": "Artist" if page == "NextSong" else None,
        "length": 200.0 if page == "NextSong" else None,
    }
    event.update(overrides)
    return event


@pytest.fixture
def cutoff():
    """Cutoff used by the synthetic fixtures."""
    return pd.Timestamp("2018-11-10")


@pytest.fixture
def events():
    """A small, fully known event log.

    - user "1": active before the cutoff, then churns after it
    - user "2": active before the cutoff, stays
    - user "3": active only *after* the cutoff
    """
    rows = [
        make_event("1", "NextSong", "2018-11-01 10:00"),
        make_event("1", "NextSong", "2018-11-02 10:00"),
        make_event("1", "Thumbs Down", "2018-11-03 10:00"),
        make_event("1", "Cancel", "2018-11-09 10:00"),
        make_event("1", CHURN_PAGE, "2018-11-15 10:00"),  # after cutoff
        make_event("2", "NextSong", "2018-11-01 11:00"),
        make_event("2", "NextSong", "2018-11-08 11:00"),
        make_event("2", "Thumbs Up", "2018-11-09 11:00"),
        make_event("3", "NextSong", "2018-11-18 12:00"),  # entirely after cutoff
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def sample_events():
    """The committed 800-user sample, skipped if it is absent."""
    if not SAMPLE_PATH.exists():
        pytest.skip(f"sample not available at {SAMPLE_PATH}")
    from churn.data import load_events

    return load_events(SAMPLE_PATH)
