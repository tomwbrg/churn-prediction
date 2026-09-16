"""Feature engineering.

``build_features`` turns a variable-length event log into one row per user
(75 features). Every aggregate is computed on events at or before the cutoff,
so nothing here can see the window the label is drawn from.
"""

from datetime import timedelta

import pandas as pd

from churn.config import LEAKY_PAGES
from churn.data import filter_before_cutoff, get_churned_users


def build_labels(df: pd.DataFrame) -> pd.Series:
    """Build binary churn labels from the event log.

    A user is churned if they have any ``Cancellation Confirmation`` event.

    Args:
        df: Raw (unfiltered) events DataFrame. Unlike the features, labels are
            read from the *full* window on purpose.

    Returns:
        Series indexed by userId, 1 = churned, 0 = retained.
    """
    churned = get_churned_users(df)
    all_users = df["userId"].unique()
    return pd.Series(
        [1 if uid in churned else 0 for uid in all_users],
        index=all_users,
        name="churned",
    )


def _page_count(page_counts: pd.DataFrame, page: str) -> pd.Series:
    """Count of one page per user, as a Series aligned on every user.

    ``page_counts.get(page, 0)`` returns a bare int when the page is absent
    from the data, which silently breaks every arithmetic expression built on
    it. A page missing from a slice is normal, so always return a Series.
    """
    if page in page_counts.columns:
        return page_counts[page]
    return pd.Series(0, index=page_counts.index, name=page)


def build_features(df, cutoff):
    """
    Build all 75 features for users from event data.
    Only uses events with time <= cutoff (temporal filtering for leakage prevention).
    Excludes leaky pages from page count features.
    """
    df_filtered = filter_before_cutoff(df, cutoff)
    features = {}
    user_groups = df_filtered.groupby("userId")

    # Session and activity aggregates
    features["total_sessions"] = user_groups["sessionId"].nunique()
    features["total_events"] = user_groups.size()
    features["songs_played"] = user_groups["song"].count()
    features["unique_artists"] = user_groups["artist"].nunique()
    features["unique_songs"] = user_groups["song"].nunique()

    features["days_active"] = user_groups["time"].apply(lambda x: (x.max() - x.min()).days + 1)
    features["days_since_registration"] = user_groups.apply(
        lambda x: (cutoff - x["registration"].iloc[0]).days, include_groups=False
    )
    features["days_since_last_activity"] = user_groups["time"].apply(
        lambda x: (cutoff - x.max()).days
    )

    # Rate features
    features["events_per_day"] = features["total_events"] / features["days_active"].clip(lower=1)
    features["sessions_per_day"] = features["total_sessions"] / features["days_active"].clip(
        lower=1
    )
    features["songs_per_day"] = features["songs_played"] / features["days_active"].clip(lower=1)
    features["songs_per_session"] = features["songs_played"] / features["total_sessions"].clip(
        lower=1
    )

    # Page counts (excluding leaky pages)
    page_counts = df_filtered.groupby(["userId", "page"]).size().unstack(fill_value=0)
    safe_pages = [p for p in page_counts.columns if p not in LEAKY_PAGES]

    for page in safe_pages:
        features["page_" + page] = page_counts[page]

    # Thumbs features
    thumbs_up = _page_count(page_counts, "Thumbs Up")
    thumbs_down = _page_count(page_counts, "Thumbs Down")
    features["thumbs_ratio"] = thumbs_up / (thumbs_up + thumbs_down + 1)
    features["thumbs_per_song"] = (thumbs_up + thumbs_down) / (features["songs_played"] + 1)

    # Diversity features
    features["song_diversity"] = features["unique_songs"] / features["songs_played"].clip(lower=1)
    features["artist_diversity"] = features["unique_artists"] / features["songs_played"].clip(
        lower=1
    )
    features["avg_song_length"] = user_groups["length"].mean()
    features["avg_items_session"] = user_groups["itemInSession"].mean()
    features["max_items_session"] = user_groups["itemInSession"].max()

    # User state features
    features["is_paid"] = user_groups["level"].apply(lambda x: (x == "paid").sum() / len(x))
    features["is_male"] = user_groups["gender"].apply(lambda x: (x == "M").sum() / len(x))

    # Problem indicators from page counts
    features["downgrades"] = _page_count(page_counts, "Downgrade")
    features["upgrades"] = _page_count(page_counts, "Upgrade")
    features["help_visits"] = _page_count(page_counts, "Help")
    features["errors"] = _page_count(page_counts, "Error")
    features["settings_visits"] = _page_count(page_counts, "Settings")

    # Activity trend (first half vs second half)
    def activity_trend(group):
        group = group.sort_values("time")
        total_days = (group["time"].max() - group["time"].min()).days + 1
        if total_days < 2:
            return 0
        mid = group["time"].min() + pd.Timedelta(days=total_days / 2)
        first = len(group[group["time"] < mid])
        second = len(group[group["time"] >= mid])
        return (second / first) - 1 if first > 0 else 1

    features["activity_trend"] = user_groups.apply(activity_trend, include_groups=False)

    # Recency features
    def recent_activity(group, days):
        cutoff_recent = group["time"].max() - pd.Timedelta(days=days)
        return len(group[group["time"] >= cutoff_recent])

    features["events_last_7d"] = user_groups.apply(
        lambda x: recent_activity(x, 7), include_groups=False
    )
    features["events_last_3d"] = user_groups.apply(
        lambda x: recent_activity(x, 3), include_groups=False
    )
    features["recent_ratio"] = features["events_last_7d"] / features["total_events"].clip(lower=1)

    # Advanced recency features
    features["days_without_song"] = user_groups.apply(
        lambda x: (cutoff - x[x["song"].notna()]["time"].max()).days
        if x["song"].notna().any()
        else 999,
        include_groups=False,
    )

    features["events_last_14d"] = user_groups.apply(
        lambda x: recent_activity(x, 14), include_groups=False
    )

    songs_7d = (
        df_filtered[df_filtered["time"] >= cutoff - timedelta(days=7)]
        .groupby("userId")["song"]
        .count()
    )
    features["songs_last_7d"] = songs_7d.reindex(features["total_events"].index, fill_value=0)

    songs_14d = (
        df_filtered[df_filtered["time"] >= cutoff - timedelta(days=14)]
        .groupby("userId")["song"]
        .count()
    )
    features["songs_last_14d"] = songs_14d.reindex(features["total_events"].index, fill_value=0)

    features["activity_acceleration"] = (features["events_last_3d"] / 3) - (
        features["events_last_7d"] / 7
    )

    old_events = features["total_events"] - features["events_last_7d"]
    features["recent_vs_old_ratio"] = features["events_last_7d"] / (old_events + 1)

    # Downgrade/upgrade status
    level_first = df_filtered.sort_values("time").groupby("userId")["level"].first()
    level_last = df_filtered.sort_values("time").groupby("userId")["level"].last()
    features["downgraded"] = ((level_first == "paid") & (level_last == "free")).astype(int)
    features["upgraded"] = ((level_first == "free") & (level_last == "paid")).astype(int)

    # Inactivity score
    features["inactivity_score"] = (
        (features["events_last_3d"] == 0).astype(int) * 3
        + (features["events_last_7d"] < 5).astype(int) * 2
        + (features["events_last_14d"] < 10).astype(int) * 1
    )

    # Frustration features
    features["error_rate"] = features["errors"] / (features["total_events"] + 1)
    features["help_rate"] = features["help_visits"] / (features["total_sessions"] + 1)
    features["frustration_score"] = features["error_rate"] + features["help_rate"]

    features["listening_intensity"] = features["songs_played"] / (features["days_active"] * 24 + 1)

    # Temporal patterns
    df_filtered["is_weekend"] = df_filtered["time"].dt.dayofweek >= 5
    weekend_events = df_filtered[df_filtered["is_weekend"]].groupby("userId").size()
    weekday_events = df_filtered[~df_filtered["is_weekend"]].groupby("userId").size()

    features["weekend_events"] = weekend_events.reindex(
        features["total_events"].index, fill_value=0
    )
    features["weekday_events"] = weekday_events.reindex(
        features["total_events"].index, fill_value=0
    )
    features["weekend_ratio"] = features["weekend_events"] / (features["total_events"] + 1)
    features["weekday_ratio"] = features["weekday_events"] / (features["total_events"] + 1)

    df_filtered["hour"] = df_filtered["time"].dt.hour
    peak_events = (
        df_filtered[(df_filtered["hour"] >= 18) & (df_filtered["hour"] <= 23)]
        .groupby("userId")
        .size()
    )
    features["peak_hour_events"] = peak_events.reindex(features["total_events"].index, fill_value=0)
    features["peak_hour_ratio"] = features["peak_hour_events"] / (features["total_events"] + 1)

    morning_events = (
        df_filtered[(df_filtered["hour"] >= 6) & (df_filtered["hour"] <= 12)]
        .groupby("userId")
        .size()
    )
    features["morning_events"] = morning_events.reindex(
        features["total_events"].index, fill_value=0
    )
    features["morning_ratio"] = features["morning_events"] / (features["total_events"] + 1)

    # Session consistency
    std_items = user_groups["itemInSession"].std()
    features["session_consistency"] = 1 / (std_items + 1)

    # Social engagement
    add_friend = _page_count(page_counts, "Add Friend")
    add_playlist = _page_count(page_counts, "Add to Playlist")
    features["social_engagement"] = (add_friend + add_playlist) / (features["songs_played"] + 1)

    # Ad exposure
    roll_advert = _page_count(page_counts, "Roll Advert")
    features["ad_exposure"] = roll_advert / (features["songs_played"] + 1)
    features["ad_per_session"] = roll_advert / (features["total_sessions"] + 1)

    # Downgrade intent
    features["downgrade_intent"] = (features["downgrades"] > 0).astype(int) * (
        1 - features["downgraded"]
    )

    # Cancel and submit downgrade features
    cancel_count = _page_count(page_counts, "Cancel")
    submit_downgrade = _page_count(page_counts, "Submit Downgrade")

    features["cancel_count"] = cancel_count
    features["submit_downgrade_count"] = submit_downgrade
    features["cancel_ratio"] = cancel_count / (features["total_events"] + 1)
    features["submit_downgrade_ratio"] = submit_downgrade / (features["total_events"] + 1)

    # Problem score
    features["problem_score"] = (
        features["help_visits"] * 1.0
        + features["errors"] * 1.5
        + features["downgrades"] * 2.0
        + cancel_count * 3.0
        + submit_downgrade * 3.0
    )

    # Artist exploration
    artist_counts = df_filtered.groupby(["userId", "artist"]).size()
    artist_variance = artist_counts.groupby("userId").std()
    features["artist_exploration"] = artist_variance.reindex(
        features["total_events"].index, fill_value=0
    )

    # Composite churn risk
    features["churn_risk_composite"] = (
        features["days_without_song"] / 100
        + features["inactivity_score"]
        + features["frustration_score"] * 10
        + (1 - features["recent_vs_old_ratio"])
        + features["cancel_ratio"] * 100
        + features["submit_downgrade_ratio"] * 100
    )

    # Strategic interaction features
    features["cancel_x_inactive"] = cancel_count * features["days_since_last_activity"]
    features["cancel_x_decline"] = cancel_count * (1 - features["recent_vs_old_ratio"])
    features["ad_x_free"] = features["ad_exposure"] * (1 - features["is_paid"])
    features["frustration_x_inactive"] = (
        features["frustration_score"] * features["inactivity_score"]
    )
    features["sessions_per_day_squared"] = features["sessions_per_day"] ** 2

    # Assemble dataframe
    features_df = pd.DataFrame(features)
    features_df["userId"] = features_df.index

    return features_df
