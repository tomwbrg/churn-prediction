"""
Feature engineering for churn prediction.

"""

import pandas as pd
import numpy as np
from datetime import timedelta

from config import LEAKY_PAGES, CHURN_INDICATOR_PAGE


def build_labels(events_df: pd.DataFrame) -> pd.Series:
    """
    Build churn labels from events data.
    
    Churned = user has a 'Cancellation Confirmation' event.
    
    Args:
        events_df: Raw events DataFrame
    
    Returns:
        Series mapping userId -> 1 (churned) or 0 (retained)
    """
    churned_users = set(
        events_df[events_df["page"] == CHURN_INDICATOR_PAGE]["userId"].unique()
    )
    
    all_users = events_df["userId"].unique()
    labels = pd.Series(
        [1 if uid in churned_users else 0 for uid in all_users],
        index=all_users,
        name="churned"
    )
    
    return labels


def build_features(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Build all 75 features for users from event data.
    
    IMPORTANT: Only uses events with time <= cutoff (anti-leakage).
    
    Args:
        df: Events DataFrame
        cutoff: Temporal cutoff - only events with time <= cutoff are used
    
    Returns:
        DataFrame with features, indexed by userId
    """
    # Temporal filtering - CRITICAL for preventing leakage
    df_filtered = df[df["time"] <= cutoff].copy()
    
    features = {}
    user_groups = df_filtered.groupby("userId")
    
    # =========================================================================
    # BASE FEATURES (44)
    # =========================================================================
    
    features["total_sessions"] = user_groups["sessionId"].nunique()
    features["total_events"] = user_groups.size()
    features["songs_played"] = user_groups["song"].count()
    features["unique_artists"] = user_groups["artist"].nunique()
    features["unique_songs"] = user_groups["song"].nunique()
    
    features["days_active"] = user_groups["time"].apply(
        lambda x: (x.max() - x.min()).days + 1
    )
    features["days_since_registration"] = user_groups.apply(
        lambda x: (cutoff - x["registration"].iloc[0]).days,
        include_groups=False
    )
    features["days_since_last_activity"] = user_groups["time"].apply(
        lambda x: (cutoff - x.max()).days
    )
    
    features["events_per_day"] = features["total_events"] / features["days_active"].clip(lower=1)
    features["sessions_per_day"] = features["total_sessions"] / features["days_active"].clip(lower=1)
    features["songs_per_day"] = features["songs_played"] / features["days_active"].clip(lower=1)
    features["songs_per_session"] = features["songs_played"] / features["total_sessions"].clip(lower=1)
    
    # Page counts (excluding leaky pages)
    page_counts = df_filtered.groupby(["userId", "page"]).size().unstack(fill_value=0)
    safe_pages = [p for p in page_counts.columns if p not in LEAKY_PAGES]
    
    for page in safe_pages:
        features[f"page_{page}"] = page_counts[page]
    
    thumbs_up = page_counts.get("Thumbs Up", pd.Series(0, index=page_counts.index))
    thumbs_down = page_counts.get("Thumbs Down", pd.Series(0, index=page_counts.index))
    features["thumbs_ratio"] = thumbs_up / (thumbs_up + thumbs_down + 1)
    features["thumbs_per_song"] = (thumbs_up + thumbs_down) / (features["songs_played"] + 1)
    
    features["song_diversity"] = features["unique_songs"] / features["songs_played"].clip(lower=1)
    features["artist_diversity"] = features["unique_artists"] / features["songs_played"].clip(lower=1)
    features["avg_song_length"] = user_groups["length"].mean()
    features["avg_items_session"] = user_groups["itemInSession"].mean()
    features["max_items_session"] = user_groups["itemInSession"].max()
    
    features["is_paid"] = user_groups["level"].apply(lambda x: (x == "paid").sum() / len(x))
    features["is_male"] = user_groups["gender"].apply(lambda x: (x == "M").sum() / len(x))
    
    features["downgrades"] = page_counts.get("Downgrade", 0)
    features["upgrades"] = page_counts.get("Upgrade", 0)
    features["help_visits"] = page_counts.get("Help", 0)
    features["errors"] = page_counts.get("Error", 0)
    features["settings_visits"] = page_counts.get("Settings", 0)
    
    # Activity trend
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
    
    # Recent activity
    def recent_activity(group, days):
        cutoff_recent = group["time"].max() - pd.Timedelta(days=days)
        return len(group[group["time"] >= cutoff_recent])
    
    features["events_last_7d"] = user_groups.apply(lambda x: recent_activity(x, 7), include_groups=False)
    features["events_last_3d"] = user_groups.apply(lambda x: recent_activity(x, 3), include_groups=False)
    features["recent_ratio"] = features["events_last_7d"] / features["total_events"].clip(lower=1)
    
    # =========================================================================
    # ADVANCED FEATURES (26)
    # =========================================================================
    
    features["days_without_song"] = user_groups.apply(
        lambda x: (cutoff - x[x["song"].notna()]["time"].max()).days
        if x["song"].notna().any() else 999,
        include_groups=False
    )
    
    features["events_last_14d"] = user_groups.apply(lambda x: recent_activity(x, 14), include_groups=False)
    
    songs_7d = df_filtered[df_filtered["time"] >= cutoff - timedelta(days=7)].groupby("userId")["song"].count()
    features["songs_last_7d"] = songs_7d.reindex(features["total_events"].index, fill_value=0)
    
    songs_14d = df_filtered[df_filtered["time"] >= cutoff - timedelta(days=14)].groupby("userId")["song"].count()
    features["songs_last_14d"] = songs_14d.reindex(features["total_events"].index, fill_value=0)
    
    features["activity_acceleration"] = (features["events_last_3d"] / 3) - (features["events_last_7d"] / 7)
    
    old_events = features["total_events"] - features["events_last_7d"]
    features["recent_vs_old_ratio"] = features["events_last_7d"] / (old_events + 1)
    
    level_first = df_filtered.sort_values("time").groupby("userId")["level"].first()
    level_last = df_filtered.sort_values("time").groupby("userId")["level"].last()
    features["downgraded"] = ((level_first == "paid") & (level_last == "free")).astype(int)
    features["upgraded"] = ((level_first == "free") & (level_last == "paid")).astype(int)
    
    features["inactivity_score"] = (
        (features["events_last_3d"] == 0).astype(int) * 3 +
        (features["events_last_7d"] < 5).astype(int) * 2 +
        (features["events_last_14d"] < 10).astype(int) * 1
    )
    
    features["error_rate"] = features["errors"] / (features["total_events"] + 1)
    features["help_rate"] = features["help_visits"] / (features["total_sessions"] + 1)
    features["frustration_score"] = features["error_rate"] + features["help_rate"]
    
    features["listening_intensity"] = features["songs_played"] / (features["days_active"] * 24 + 1)
    
    df_filtered["is_weekend"] = df_filtered["time"].dt.dayofweek >= 5
    weekend_events = df_filtered[df_filtered["is_weekend"]].groupby("userId").size()
    weekday_events = df_filtered[~df_filtered["is_weekend"]].groupby("userId").size()
    
    features["weekend_events"] = weekend_events.reindex(features["total_events"].index, fill_value=0)
    features["weekday_events"] = weekday_events.reindex(features["total_events"].index, fill_value=0)
    features["weekend_ratio"] = features["weekend_events"] / (features["total_events"] + 1)
    features["weekday_ratio"] = features["weekday_events"] / (features["total_events"] + 1)
    
    df_filtered["hour"] = df_filtered["time"].dt.hour
    peak_events = df_filtered[(df_filtered["hour"] >= 18) & (df_filtered["hour"] <= 23)].groupby("userId").size()
    features["peak_hour_events"] = peak_events.reindex(features["total_events"].index, fill_value=0)
    features["peak_hour_ratio"] = features["peak_hour_events"] / (features["total_events"] + 1)
    
    morning_events = df_filtered[(df_filtered["hour"] >= 6) & (df_filtered["hour"] <= 12)].groupby("userId").size()
    features["morning_events"] = morning_events.reindex(features["total_events"].index, fill_value=0)
    features["morning_ratio"] = features["morning_events"] / (features["total_events"] + 1)
    
    std_items = user_groups["itemInSession"].std()
    features["session_consistency"] = 1 / (std_items + 1)
    
    add_friend = page_counts.get("Add Friend", pd.Series(0, index=page_counts.index))
    add_playlist = page_counts.get("Add to Playlist", pd.Series(0, index=page_counts.index))
    features["social_engagement"] = (add_friend + add_playlist) / (features["songs_played"] + 1)
    
    roll_advert = page_counts.get("Roll Advert", pd.Series(0, index=page_counts.index))
    features["ad_exposure"] = roll_advert / (features["songs_played"] + 1)
    features["ad_per_session"] = roll_advert / (features["total_sessions"] + 1)
    
    features["downgrade_intent"] = (features["downgrades"] > 0).astype(int) * (1 - features["downgraded"])
    
    cancel_count = page_counts.get("Cancel", pd.Series(0, index=page_counts.index))
    submit_downgrade = page_counts.get("Submit Downgrade", pd.Series(0, index=page_counts.index))
    
    features["cancel_count"] = cancel_count
    features["submit_downgrade_count"] = submit_downgrade
    features["cancel_ratio"] = cancel_count / (features["total_events"] + 1)
    features["submit_downgrade_ratio"] = submit_downgrade / (features["total_events"] + 1)
    
    features["problem_score"] = (
        features["help_visits"] * 1.0 +
        features["errors"] * 1.5 +
        features["downgrades"] * 2.0 +
        cancel_count * 3.0 +
        submit_downgrade * 3.0
    )
    
    artist_counts = df_filtered.groupby(["userId", "artist"]).size()
    artist_variance = artist_counts.groupby("userId").std()
    features["artist_exploration"] = artist_variance.reindex(features["total_events"].index, fill_value=0)
    
    features["churn_risk_composite"] = (
        features["days_without_song"] / 100 +
        features["inactivity_score"] +
        features["frustration_score"] * 10 +
        (1 - features["recent_vs_old_ratio"]) +
        features["cancel_ratio"] * 100 +
        features["submit_downgrade_ratio"] * 100
    )
    
    # =========================================================================
    # STRATEGIC INTERACTIONS (5)
    # =========================================================================
    
    features["cancel_x_inactive"] = cancel_count * features["days_since_last_activity"]
    features["cancel_x_decline"] = cancel_count * (1 - features["recent_vs_old_ratio"])
    features["ad_x_free"] = features["ad_exposure"] * (1 - features["is_paid"])
    features["frustration_x_inactive"] = features["frustration_score"] * features["inactivity_score"]
    features["sessions_per_day_squared"] = features["sessions_per_day"] ** 2
    
    # Assemble DataFrame
    features_df = pd.DataFrame(features)
    features_df["userId"] = features_df.index
    
    return features_df

