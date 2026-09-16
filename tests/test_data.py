"""Tests for data importing and filtering."""

import pandas as pd
import pytest

from churn.config import CHURN_PAGE, REQUIRED_COLUMNS
from churn.data import (
    SchemaError,
    drop_leaky_pages,
    filter_before_cutoff,
    filter_valid_users,
    get_churned_users,
    load_events,
    normalize_types,
    validate_schema,
)

from .conftest import make_event


class TestValidateSchema:
    def test_accepts_a_complete_frame(self, events):
        assert validate_schema(events) is events

    @pytest.mark.parametrize("missing", ["userId", "page", "time", "registration"])
    def test_rejects_a_missing_column(self, events, missing):
        with pytest.raises(SchemaError, match=missing):
            validate_schema(events.drop(columns=[missing]))

    def test_error_names_every_missing_column(self, events):
        with pytest.raises(SchemaError) as excinfo:
            validate_schema(events.drop(columns=["page", "time"]))
        assert "page" in str(excinfo.value)
        assert "time" in str(excinfo.value)

    def test_sample_file_satisfies_the_schema(self, sample_events):
        for column in REQUIRED_COLUMNS:
            assert column in sample_events.columns


class TestNormalizeTypes:
    def test_parses_string_timestamps(self, events):
        raw = events.assign(time=events["time"].astype(str))
        out = normalize_types(raw)
        assert pd.api.types.is_datetime64_any_dtype(out["time"])

    def test_casts_user_id_to_string(self, events):
        out = normalize_types(events.assign(userId=[1] * len(events)))
        assert out["userId"].map(type).eq(str).all()

    def test_does_not_mutate_the_input(self, events):
        raw = events.assign(time=events["time"].astype(str))
        normalize_types(raw)
        assert raw["time"].dtype == object


class TestLoadEvents:
    def test_missing_file_points_to_the_download_script(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download_data"):
            load_events(tmp_path / "absent.parquet")

    def test_roundtrip_preserves_rows(self, events, tmp_path):
        path = tmp_path / "events.parquet"
        events.to_parquet(path, index=False)
        assert len(load_events(path)) == len(events)

    def test_rejects_a_file_with_a_broken_schema(self, events, tmp_path):
        path = tmp_path / "broken.parquet"
        events.drop(columns=["page"]).to_parquet(path, index=False)
        with pytest.raises(SchemaError):
            load_events(path)


class TestFilterBeforeCutoff:
    """The pipeline's main defence against temporal leakage."""

    def test_drops_every_event_after_the_cutoff(self, events, cutoff):
        out = filter_before_cutoff(events, cutoff)
        assert (out["time"] <= cutoff).all()

    def test_keeps_an_event_exactly_on_the_cutoff(self, cutoff):
        df = pd.DataFrame([make_event("1", "NextSong", cutoff)])
        assert len(filter_before_cutoff(df, cutoff)) == 1

    def test_drops_a_user_active_only_after_the_cutoff(self, events, cutoff):
        out = filter_before_cutoff(events, cutoff)
        assert "3" not in set(out["userId"])

    def test_does_not_mutate_the_input(self, events, cutoff):
        before = len(events)
        filter_before_cutoff(events, cutoff)
        assert len(events) == before

    def test_an_early_cutoff_empties_the_frame(self, events):
        assert filter_before_cutoff(events, pd.Timestamp("2018-01-01")).empty


class TestDropLeakyPages:
    def test_removes_the_churn_confirmation_event(self, events):
        assert CHURN_PAGE not in set(drop_leaky_pages(events)["page"])

    def test_keeps_every_other_page(self, events):
        out = drop_leaky_pages(events)
        assert len(out) == len(events[events["page"] != CHURN_PAGE])

    def test_keeps_the_cancel_page(self, events):
        """'Cancel' is a visit to the cancellation page, not the act itself:
        it is a legitimate predictor and must survive."""
        assert "Cancel" in set(drop_leaky_pages(events)["page"])


class TestFilterValidUsers:
    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_drops_rows_without_a_usable_user_id(self, events, bad):
        polluted = pd.concat(
            [events, pd.DataFrame([make_event(bad, "Home", "2018-11-05")])],
            ignore_index=True,
        )
        out = filter_valid_users(polluted)
        assert len(out) == len(events)

    def test_keeps_every_valid_row(self, events):
        assert len(filter_valid_users(events)) == len(events)


class TestGetChurnedUsers:
    def test_finds_the_user_who_confirmed_a_cancellation(self, events):
        assert get_churned_users(events) == {"1"}

    def test_returns_empty_when_nobody_churned(self, events):
        assert get_churned_users(drop_leaky_pages(events)) == set()

    def test_sample_churn_rate_is_realistic(self, sample_events):
        """The committed sample is stratified: it must mirror the full
        dataset's 22.3% churn rate, not a random-row sample's noise."""
        rate = len(get_churned_users(sample_events)) / sample_events["userId"].nunique()
        assert 0.18 < rate < 0.27
