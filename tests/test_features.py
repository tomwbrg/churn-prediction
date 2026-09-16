"""Tests for label construction and feature engineering.

The leakage tests are the important ones: a model that sees the future scores
beautifully in validation and is worthless in production, and nothing in the
metrics reveals it. These assertions are what make the cutoff trustworthy.
"""

import pandas as pd
import pytest

from churn.config import CHURN_PAGE
from churn.features import build_features, build_labels

from .conftest import make_event


class TestBuildLabels:
    def test_labels_the_user_who_confirmed_a_cancellation(self, events):
        assert build_labels(events)["1"] == 1

    def test_labels_the_retained_user_as_zero(self, events):
        assert build_labels(events)["2"] == 0

    def test_covers_every_user_exactly_once(self, events):
        labels = build_labels(events)
        assert len(labels) == events["userId"].nunique()
        assert labels.index.is_unique

    def test_is_binary(self, sample_events):
        assert set(build_labels(sample_events).unique()) <= {0, 1}

    def test_a_cancel_visit_alone_is_not_churn(self):
        """Visiting the cancellation page without confirming is not churn."""
        df = pd.DataFrame([make_event("9", "Cancel", "2018-11-02")])
        assert build_labels(df)["9"] == 0


class TestFeatureLeakage:
    def test_no_feature_is_derived_from_the_churn_page(self, events, cutoff):
        out = build_features(events, cutoff)
        assert not [c for c in out.columns if CHURN_PAGE in c]

    def test_events_after_the_cutoff_do_not_change_any_feature(self, events, cutoff):
        """The decisive test: replaying the same history with extra future
        activity must produce byte-identical features."""
        baseline = build_features(events, cutoff)

        future = pd.concat(
            [
                events,
                pd.DataFrame(
                    [
                        make_event("1", "NextSong", "2018-11-12 10:00"),
                        make_event("1", "Thumbs Up", "2018-11-13 10:00"),
                        make_event("2", CHURN_PAGE, "2018-11-14 11:00"),
                    ]
                ),
            ],
            ignore_index=True,
        )
        with_future = build_features(future, cutoff)

        common = baseline.index.intersection(with_future.index)
        pd.testing.assert_frame_equal(
            baseline.loc[common].sort_index(axis=1),
            with_future.loc[common].sort_index(axis=1),
        )

    def test_a_user_with_no_history_before_the_cutoff_is_excluded(self, events, cutoff):
        assert "3" not in set(build_features(events, cutoff)["userId"])

    def test_an_earlier_cutoff_sees_strictly_less_activity(self, events):
        late = build_features(events, pd.Timestamp("2018-11-10"))
        early = build_features(events, pd.Timestamp("2018-11-02"))
        assert early.loc["1", "total_events"] < late.loc["1", "total_events"]


class TestBuildFeatures:
    def test_returns_one_row_per_user(self, events, cutoff):
        out = build_features(events, cutoff)
        assert len(out) == out["userId"].nunique()

    def test_counts_events_correctly(self, events, cutoff):
        """User 1 has 4 events at or before the cutoff, user 2 has 3."""
        out = build_features(events, cutoff)
        assert out.loc["1", "total_events"] == 4
        assert out.loc["2", "total_events"] == 3

    def test_measures_recency_from_the_cutoff(self, events, cutoff):
        """Recency is a truncated day count, not a rounded one.

        User 1's last event is 2018-11-09 10:00, 14 hours before the cutoff,
        which counts as 0 whole days. Pinning this down matters: the feature
        feeds inactivity_score and churn_risk_composite.
        """
        out = build_features(events, cutoff)
        assert out.loc["1", "days_since_last_activity"] == 0

        earlier = pd.DataFrame(
            [
                make_event("1", "NextSong", "2018-11-01 10:00"),
                make_event("1", "NextSong", "2018-11-08 10:00"),
            ]
        )
        assert build_features(earlier, cutoff).loc["1", "days_since_last_activity"] == 1

    def test_counts_cancellation_page_visits(self, events, cutoff):
        out = build_features(events, cutoff)
        assert out.loc["1", "cancel_count"] == 1
        assert out.loc["2", "cancel_count"] == 0

    def test_is_deterministic(self, events, cutoff):
        pd.testing.assert_frame_equal(
            build_features(events, cutoff), build_features(events, cutoff)
        )

    def test_produces_no_missing_user_ids(self, sample_events, cutoff):
        out = build_features(sample_events, cutoff)
        assert out["userId"].notna().all()

    @pytest.mark.parametrize(
        "feature",
        ["total_events", "total_sessions", "songs_played", "days_active"],
    )
    def test_count_features_are_never_negative(self, sample_events, cutoff, feature):
        assert (build_features(sample_events, cutoff)[feature] >= 0).all()

    def test_ratio_features_stay_within_bounds(self, sample_events, cutoff):
        out = build_features(sample_events, cutoff)
        for feature in ("thumbs_ratio", "is_paid", "is_male", "weekend_ratio"):
            assert out[feature].between(0, 1).all(), feature
