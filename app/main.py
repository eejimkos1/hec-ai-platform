"""HEC AI Platform — Streamlit entry point.

Run with:
    streamlit run app/main.py
"""

import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Ensure the project root is on sys.path so that `app.config` resolves
# correctly whether the script is run as `streamlit run app/main.py`
# (which injects the `app/` directory) or as a module from the project root.
_HERE = Path(__file__).resolve().parent          # …/meli/app
_ROOT = _HERE.parent                              # …/meli
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
from app.config import (
    PAGE_TITLE,
    LAYOUT,
    PIPELINE_STEPS_5,
    DATA_FEATURES_DIR,
    DATA_RAW_DIR,
    COLORS,
    EUR_PER_YIELD_PCT,
    AVG_BATCHES_PER_YEAR,
    check_data_status,
)
from app.components.theme import inject_hec_css, render_money_callout, apply_plotly_theme

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="⚓",
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Auto-build pipeline on first run (for Streamlit Cloud deployment)
# ---------------------------------------------------------------------------
from run_pipeline import pipeline_needed, run_pipeline  # noqa: E402

if pipeline_needed():
    with st.spinner("Building data pipeline (first run only — ~2 min)..."):
        run_pipeline()
    st.cache_data.clear()

inject_hec_css()

# ---------------------------------------------------------------------------
# Data loaders (cached so re-runs don't re-read CSV files)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_separation_features() -> pd.DataFrame | None:
    path = DATA_FEATURES_DIR / "separation_features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["processing_date"])
    return df


@st.cache_data(show_spinner=False)
def load_fleet_features() -> pd.DataFrame | None:
    path = DATA_FEATURES_DIR / "fleet_features.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_voyage_history() -> pd.DataFrame | None:
    path = DATA_RAW_DIR / "voyage_history.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["departure_time"])
    return df


@st.cache_data(show_spinner=False)
def load_fleet_status() -> pd.DataFrame | None:
    path = DATA_RAW_DIR / "fleet_status.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("**HEC**")
    st.caption("Hellenic Environmental Center")
    st.markdown("**AI-Powered Operations Platform**")
    st.divider()
    st.markdown(
        "Optimizing petroleum waste separation & fleet logistics through machine learning."
    )
    st.divider()

    st.subheader("Data Status")
    status = check_data_status()

    raw_ready = sum(status["raw"].values())
    raw_total = len(status["raw"])
    feat_ready = sum(status["features"].values())
    feat_total = len(status["features"])
    model_ready = sum(status["models"].values())
    model_total = len(status["models"])

    icon_raw = "✅" if raw_ready == raw_total else ("⚠️" if raw_ready > 0 else "❌")
    icon_feat = "✅" if feat_ready == feat_total else ("⚠️" if feat_ready > 0 else "❌")
    icon_mod = "✅" if model_ready == model_total else ("⚠️" if model_ready > 0 else "❌")

    st.markdown(f"{icon_raw} Raw data: {raw_ready}/{raw_total} files")
    st.markdown(f"{icon_feat} Features: {feat_ready}/{feat_total} files")
    st.markdown(f"{icon_mod} Models: {model_ready}/{model_total} files")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

sep_df = load_separation_features()
fleet_df = load_fleet_features()
voyage_df = load_voyage_history()
fleet_status_df = load_fleet_status()

data_missing = sep_df is None or voyage_df is None

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("HEC AI Platform — Operations Intelligence")
st.markdown(
    "This platform applies machine learning to HEC's core operations: petroleum waste separation "
    "process control and fleet collection logistics. "
    "Every view follows the pipeline: "
    "**Source Data → Analytics → Feature Engineering → ML Model → Results**."
)

st.divider()

# ---------------------------------------------------------------------------
# Financial Impact Summary
# ---------------------------------------------------------------------------

st.header("Financial Impact Summary")

if data_missing:
    st.info(
        "Financial metrics will populate here once data files are generated. "
        "See pipeline instructions below."
    )
else:
    # --- Separation economics ---
    total_batches = len(sep_df)
    avg_yield = sep_df["oil_recovery_yield_pct"].mean()
    avg_margin = sep_df["net_margin_eur"].mean()
    quality_pass_rate = sep_df["quality_pass"].mean() * 100

    # --- Fleet economics ---
    total_voyages = len(voyage_df)
    total_revenue = voyage_df["revenue_eur"].sum() if "revenue_eur" in voyage_df.columns else 0.0
    avg_voyage_margin = voyage_df["voyage_margin_eur"].mean()
    if "current_cargo_pct" in (fleet_status_df.columns if fleet_status_df is not None else []):
        fleet_util = fleet_status_df["current_cargo_pct"].mean()
        util_str = f"{fleet_util:.1f}%"
    else:
        util_str = "N/A"

    # --- AI opportunity ---
    yield_improvement_pp = 1.07
    annual_separation_savings = yield_improvement_pp * EUR_PER_YIELD_PCT * AVG_BATCHES_PER_YEAR
    if "fuel_cost_eur" in voyage_df.columns:
        annual_fleet_savings = voyage_df["fuel_cost_eur"].sum() / 4.5 * 0.10
    else:
        annual_fleet_savings = 0.0
    total_ai_opportunity = annual_separation_savings + annual_fleet_savings

    render_money_callout(
        title="Estimated Annual AI Optimization Opportunity",
        amount_eur=total_ai_opportunity,
        description="Combining separation yield improvement and fleet route optimization",
    )

    col_sep, col_fleet = st.columns(2)

    with col_sep:
        st.subheader("Separation Economics")
        m1, m2 = st.columns(2)
        m1.metric("Total Batches Processed", f"{total_batches:,}")
        m2.metric("Avg Oil Recovery Yield", f"{avg_yield:.1f}%")
        m3, m4 = st.columns(2)
        m3.metric("Avg Net Margin / Batch", f"€{avg_margin:,.0f}")
        m4.metric("Quality Pass Rate", f"{quality_pass_rate:.1f}%")
        st.markdown(
            f"<span style='font-size:0.85rem;color:{COLORS['text_muted']};'>"
            "A 1 pp yield improvement = €1,500/batch = €18M/year at current throughput."
            "</span>",
            unsafe_allow_html=True,
        )

    with col_fleet:
        st.subheader("Fleet Economics")
        m1, m2 = st.columns(2)
        m1.metric("Total Voyages", f"{total_voyages:,}")
        m2.metric("Fleet Revenue", f"€{total_revenue:,.0f}")
        m3, m4 = st.columns(2)
        m3.metric("Avg Voyage Margin", f"€{avg_voyage_margin:,.0f}")
        m4.metric("Fleet Utilisation", util_str)
        st.markdown(
            f"<span style='font-size:0.85rem;color:{COLORS['text_muted']};'>"
            "Route optimization reduces fuel costs by 8–15% per voyage."
            "</span>",
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Monthly Revenue Trend
# ---------------------------------------------------------------------------

st.header("Monthly Revenue Trend")

if data_missing:
    st.info("Monthly trend chart will appear once data files are generated.")
else:
    try:
        sep_monthly = (
            sep_df.dropna(subset=["processing_date"])
            .set_index("processing_date")
            .resample("ME")["net_margin_eur"]
            .sum()
            .reset_index()
        )
        fleet_monthly = (
            voyage_df.dropna(subset=["departure_time"])
            .set_index("departure_time")
            .resample("ME")["voyage_margin_eur"]
            .sum()
            .reset_index()
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sep_monthly["processing_date"],
            y=sep_monthly["net_margin_eur"],
            name="Separation Margin",
            line=dict(color=COLORS["primary"], width=2),
            mode="lines+markers",
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=fleet_monthly["departure_time"],
            y=fleet_monthly["voyage_margin_eur"],
            name="Fleet Margin",
            line=dict(color=COLORS["accent"], width=2),
            mode="lines+markers",
            marker=dict(size=5),
        ))
        fig.update_xaxes(rangeslider_visible=True, title_text="Month")
        fig.update_yaxes(title_text="Margin (EUR)", tickprefix="€", separatethousands=True)
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420,
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Monthly trend chart unavailable: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Pipeline Architecture
# ---------------------------------------------------------------------------

st.header("Pipeline Architecture")
st.markdown(
    "Every use case in this platform follows the same five-step flow — from raw operational "
    "data through to actionable decisions and measurable business impact."
)

NAVY = COLORS["primary"]
TEAL = COLORS["accent"]

cols = st.columns(len(PIPELINE_STEPS_5))
for i, (col, step) in enumerate(zip(cols, PIPELINE_STEPS_5)):
    border_color = TEAL if i == 0 else NAVY
    with col:
        st.markdown(
            f"""
            <div style="
                border: 1px solid {COLORS['border']};
                border-top: 4px solid {border_color};
                border-radius: 8px;
                padding: 16px 12px;
                text-align: center;
                background: {COLORS['bg_light']};
                min-height: 120px;
            ">
                <div style="font-size: 1.8rem;">{step['icon']}</div>
                <div style="font-weight: 600; margin: 6px 0 4px; color:{NAVY}; font-size:0.9rem;">{step['label']}</div>
                <div style="font-size: 0.8rem; color: {COLORS['text_muted']};">{step['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Connector arrows between pipeline steps
arrow_cols = st.columns(len(PIPELINE_STEPS_5))
for i, col in enumerate(arrow_cols):
    if i < len(PIPELINE_STEPS_5) - 1:
        col.markdown(
            f"<div style='text-align:right; font-size:1.4rem; color:{TEAL}; margin-top:-4px;'>→</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Data generation instructions (shown only when data is missing)
# ---------------------------------------------------------------------------

if data_missing:
    st.divider()
    st.info(
        "**Data files are not yet generated.** "
        "Run the pipeline scripts below to create all required data, features, and models."
    )
    st.code(
        "python scripts/generate_data.py && "
        "python scripts/engineer_features.py && "
        "python scripts/train_models.py",
        language="bash",
    )
    st.markdown(
        "Once complete, refresh this page and all KPI metrics and charts will populate."
    )
