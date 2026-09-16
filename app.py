"""Streamlit dashboard for the churn prediction model.

    streamlit run app.py

Runs on the committed 800-user sample by default, so a fresh clone works with
no extra data. If the full dataset is present it can be selected in the sidebar.
"""

import altair as alt
import pandas as pd
import streamlit as st

from churn.config import (
    CHURN_PAGE,
    FINAL_CUTOFF,
    MODEL_PATH,
    SAMPLE_PATH,
    TRAIN_CUTOFF,
    TRAIN_PATH,
)
from churn.data import filter_before_cutoff, get_churned_users, load_events
from churn.features import build_features, build_labels
from churn.model import load_model, predict

# Categorical slots 1 and 2 of the reference palette, validated for
# colour-vision deficiency (worst-pair CVD deltaE 24.7, normal-vision 33.6).
RETAINED = "#2a78d6"
CHURNED = "#eb6834"
GRID = "#d8d8d4"

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CACHED LOADERS
# =============================================================================
# Reading 17.5M events and rebuilding 75 features costs minutes; caching on the
# path and the cutoff means it happens once per distinct input, not per rerun.


@st.cache_data(show_spinner="Loading events...")
def get_events(path_str: str) -> pd.DataFrame:
    return load_events(path_str)


@st.cache_data(show_spinner="Building features...")
def get_features(path_str: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    return build_features(get_events(path_str), cutoff)


@st.cache_data
def get_labels(path_str: str) -> pd.Series:
    return build_labels(get_events(path_str))


@st.cache_resource(show_spinner="Loading model...")
def get_model() -> dict:
    return load_model()


@alt.theme.register("churn", enable=True)
def churn_theme() -> alt.theme.ThemeConfig:
    """Recessive grid and axes, no chart border, readable label sizes.

    Registered once as a theme so every chart inherits it: Altair's
    ``configure_*`` methods only apply to a top-level chart and raise as soon
    as a chart is layered or concatenated.
    """
    return {
        "config": {
            "axis": {
                "grid": True,
                "gridColor": GRID,
                "gridOpacity": 0.6,
                "domainColor": GRID,
                "tickColor": GRID,
                "labelFontSize": 12,
                "titleFontSize": 12,
                "titleFontWeight": "normal",
            },
            "view": {"strokeWidth": 0, "continuousHeight": 280},
            "legend": {
                "labelFontSize": 12,
                "titleFontSize": 12,
                "titleFontWeight": "normal",
            },
        }
    }


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🎧 Churn Prediction")
st.sidebar.caption("Music streaming service — behavioural event logs")

sources = {"Sample (800 users)": SAMPLE_PATH}
if TRAIN_PATH.exists():
    sources["Full dataset (19,140 users)"] = TRAIN_PATH

source_label = st.sidebar.selectbox("Data source", list(sources))
data_path = str(sources[source_label])

cutoff = pd.Timestamp(
    st.sidebar.date_input(
        "Feature cutoff",
        value=TRAIN_CUTOFF.date(),
        min_value=pd.Timestamp("2018-10-05").date(),
        max_value=FINAL_CUTOFF.date(),
        help="Features use only events at or before this date. Everything after "
        "it is the window the label is drawn from — moving this line is what "
        "separates an honest model from a leaking one.",
    )
)

st.sidebar.divider()
st.sidebar.caption(
    f"Cutoff **{cutoff.date()}** · labels read from the full window.\n\n"
    "Source: `src/churn/` — the same functions the test suite covers."
)

events = get_events(data_path)
features_df = get_features(data_path, cutoff)
labels = get_labels(data_path)
churned_users = get_churned_users(events)

tab_overview, tab_predict, tab_model = st.tabs(["Overview", "Predict", "Model"])


# =============================================================================
# TAB 1 — OVERVIEW
# =============================================================================

with tab_overview:
    st.subheader("Dataset at a glance")

    n_users = events["userId"].nunique()
    churn_rate = len(churned_users) / n_users if n_users else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", f"{len(events):,}")
    c2.metric("Users", f"{n_users:,}")
    c3.metric("Churn rate", f"{churn_rate:.1%}")
    c4.metric("Events / user", f"{len(events) / max(n_users, 1):,.0f}")

    st.caption(
        f"Observation window {events['time'].min().date()} → "
        f"{events['time'].max().date()}. A user counts as churned once they "
        f"trigger a *{CHURN_PAGE}* event."
    )

    st.divider()

    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Daily activity, churned vs retained users**")
        daily = filter_before_cutoff(events, cutoff).copy()
        daily["group"] = (
            daily["userId"].isin(churned_users).map({True: "Churned", False: "Retained"})
        )
        per_day = (
            daily.assign(day=daily["time"].dt.date)
            .groupby(["day", "group"], as_index=False)
            .size()
            .rename(columns={"size": "events"})
        )
        # Churned users are a minority; raw counts would hide their trend
        # entirely, so each group is normalised by its own user count.
        sizes = daily.groupby("group")["userId"].nunique()
        per_day["events_per_user"] = per_day["events"] / per_day["group"].map(sizes)

        chart = (
            alt.Chart(per_day)
            .mark_line(strokeWidth=2, interpolate="monotone")
            .encode(
                x=alt.X("day:T", title=None),
                y=alt.Y("events_per_user:Q", title="Events per user"),
                color=alt.Color(
                    "group:N",
                    title=None,
                    scale=alt.Scale(domain=["Retained", "Churned"], range=[RETAINED, CHURNED]),
                    legend=alt.Legend(orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("day:T", title="Day"),
                    alt.Tooltip("group:N", title="Group"),
                    alt.Tooltip("events_per_user:Q", title="Events/user", format=".1f"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, width="stretch")
        st.caption(
            "Churned users trail off before they cancel — that decay is what "
            "`activity_trend` and `recent_vs_old_ratio` encode."
        )

    with right:
        st.markdown("**Most frequent page events**")
        pages = (
            filter_before_cutoff(events, cutoff)["page"]
            .value_counts()
            .head(10)
            .rename_axis("page")
            .reset_index(name="count")
        )
        page_chart = (
            alt.Chart(pages)
            .mark_bar(cornerRadiusEnd=4, color=RETAINED, size=16)
            .encode(
                x=alt.X("count:Q", title=None),
                y=alt.Y("page:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("page:N", title="Page"),
                    alt.Tooltip("count:Q", title="Events", format=","),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(page_chart, width="stretch")

    st.divider()
    st.markdown("**Feature distribution by outcome**")

    numeric = [
        c
        for c in (
            "days_since_last_activity",
            "total_events",
            "cancel_count",
            "thumbs_ratio",
            "inactivity_score",
            "songs_per_session",
        )
        if c in features_df.columns
    ]
    feature = st.selectbox("Feature", numeric, index=0)

    dist = features_df[[feature]].copy()
    dist["group"] = dist.index.map(lambda u: u in churned_users).map(
        {True: "Churned", False: "Retained"}
    )
    hist = (
        alt.Chart(dist)
        .mark_bar(opacity=0.75, cornerRadiusEnd=2)
        .encode(
            x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=40), title=feature),
            y=alt.Y("count()", title="Users", stack=None),
            color=alt.Color(
                "group:N",
                title=None,
                scale=alt.Scale(domain=["Retained", "Churned"], range=[RETAINED, CHURNED]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[alt.Tooltip("count()", title="Users")],
        )
        .properties(height=260)
    )
    st.altair_chart(hist, width="stretch")

    with st.expander("Raw events"):
        st.dataframe(events.head(200), width="stretch")


# =============================================================================
# TAB 2 — PREDICT
# =============================================================================

with tab_predict:
    st.subheader("Score a user")

    if not MODEL_PATH.exists():
        st.warning(
            "No trained model found. Run `python scripts/train.py --sample` " "to create one."
        )
    else:
        bundle = get_model()
        feature_names = bundle["feature_names"]

        user_ids = sorted(features_df["userId"].astype(str))
        col_pick, col_score = st.columns([2, 3])

        with col_pick:
            user_id = st.selectbox("User", user_ids, index=0)
            actual = "Churned" if user_id in churned_users else "Retained"
            st.caption(f"Ground truth for this user: **{actual}**")

        row = features_df.loc[features_df["userId"].astype(str) == user_id]
        X = row.reindex(columns=feature_names, fill_value=0).fillna(0)
        proba = float(predict(bundle["models"], X)[0])

        with col_score:
            st.metric("Churn probability", f"{proba:.1%}")
            st.progress(min(proba, 1.0))
            verdict = "at risk" if proba >= 0.5 else "likely to stay"
            st.caption(
                f"The model puts this user **{verdict}**. Threshold shown at 50% "
                "for readability; the submission pipeline uses a percentile cut."
            )

        st.divider()
        st.markdown("**What drives this score**")

        importance = pd.DataFrame(bundle["importance"]).head(12)
        contrib = importance.merge(
            row.melt(var_name="feature", value_name="value"), on="feature", how="left"
        )
        contrib["value"] = pd.to_numeric(contrib["value"], errors="coerce").fillna(0)

        drivers = (
            alt.Chart(contrib)
            .mark_bar(cornerRadiusEnd=4, color=CHURNED, size=14)
            .encode(
                x=alt.X("importance:Q", title="Model importance (gain)"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("value:Q", title="This user", format=".2f"),
                    alt.Tooltip("importance:Q", title="Importance", format=".0f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(drivers, width="stretch")
        st.caption(
            "Bars are the model's global importance; hover to read this user's "
            "own value for each feature."
        )

        with st.expander("All features for this user"):
            # userId is dropped before transposing: mixing a string id into a
            # column of 86 floats makes Arrow reject the whole frame.
            detail = row.drop(columns=["userId"]).T.astype(float)
            detail.columns = ["value"]
            st.dataframe(detail, width="stretch")


# =============================================================================
# TAB 3 — MODEL
# =============================================================================

with tab_model:
    st.subheader("Model performance")

    if not MODEL_PATH.exists():
        st.warning("No trained model found. Run `python scripts/train.py`.")
    else:
        bundle = get_model()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("OOF ROC-AUC", f"{bundle['oof_auc']:.4f}")
        m2.metric("CV folds", bundle["folds"])
        m3.metric("Trained on", bundle["trained_on"])
        m4.metric("Users", f"{bundle['n_users']:,}")

        st.caption(
            f"LightGBM, {bundle['folds']}-fold stratified CV, seed {bundle['seed']}, "
            f"features cut at {pd.Timestamp(bundle['train_cutoff']).date()}."
        )

        st.divider()
        left, right = st.columns(2)

        with left:
            st.markdown("**ROC-AUC per fold**")
            folds = pd.DataFrame(
                {
                    "fold": [f"{i + 1}" for i in range(len(bundle["fold_aucs"]))],
                    "auc": bundle["fold_aucs"],
                }
            )
            lo = min(folds["auc"]) - 0.02
            fold_chart = (
                alt.Chart(folds)
                .mark_bar(cornerRadiusEnd=4, color=RETAINED, size=22)
                .encode(
                    x=alt.X("fold:N", title="Fold"),
                    y=alt.Y(
                        "auc:Q",
                        title="ROC-AUC",
                        scale=alt.Scale(domain=[max(lo, 0), 1.0]),
                    ),
                    tooltip=[alt.Tooltip("auc:Q", title="AUC", format=".4f")],
                )
                .properties(height=300)
            )
            st.altair_chart(fold_chart, width="stretch")
            spread = max(bundle["fold_aucs"]) - min(bundle["fold_aucs"])
            st.caption(
                f"Spread across folds: {spread:.4f} — low variance means the "
                "score is not one lucky split."
            )

        with right:
            st.markdown("**Top 20 features by importance**")
            imp = pd.DataFrame(bundle["importance"]).head(20)
            imp_chart = (
                alt.Chart(imp)
                .mark_bar(cornerRadiusEnd=4, color=RETAINED, size=12)
                .encode(
                    x=alt.X("importance:Q", title="Average gain"),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    tooltip=[
                        alt.Tooltip("feature:N", title="Feature"),
                        alt.Tooltip("importance:Q", title="Gain", format=".0f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(imp_chart, width="stretch")

        st.divider()
        with st.expander("Full feature importance table"):
            st.dataframe(pd.DataFrame(bundle["importance"]), width="stretch")
