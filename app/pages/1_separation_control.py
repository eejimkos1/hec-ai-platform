"""Use Case 1: AI-Optimized Separation Process Control.

5-tab Streamlit page mirroring the ML pipeline:
  Tab 1 — Source Data Explorer        (financial KPIs + cost breakdown donut)
  Tab 2 — Analytics                   (NEW — interactive descriptive analytics)
  Tab 3 — Feature Engineering         (feature matrix, correlations, distributions)
  Tab 4 — ML Model Performance        (yield regressor + quality classifier)
  Tab 5 — Optimization Engine         (parameter optimizer + financial enrichment)

Run with:
    streamlit run app/main.py
"""

import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# ── project-root on sys.path so `app.*` imports resolve correctly ──────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import (
    DATA_RAW_DIR,
    DATA_FEATURES_DIR,
    MODELS_DIR,
    COLORS,
    SEPARATION_COLUMN_DESCRIPTIONS,
)
from app.components.pipeline_viewer import (
    render_pipeline_steps,
    render_data_flow_explanation,
)
from app.components.charts import (
    yield_distribution_chart,
    feature_importance_chart,
    actual_vs_predicted_chart,
    residual_distribution_chart,
    correlation_heatmap,
    waste_volume_by_facility_chart,
    waste_category_pie_chart,
    confusion_matrix_chart,
    optimization_comparison_chart,
    violin_by_group,
    time_series_with_trend,
    radar_chart,
    scatter_with_trend,
    cost_breakdown_donut,
    profitability_distribution,
)
from app.components.theme import (
    inject_hec_css,
    render_money_callout,
    apply_plotly_theme,
)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UC1 — Separation Control | HEC AI Platform",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── HEC branding ─────────────────────────────────────────────────────────────
inject_hec_css()


# ============================================================================
# Data / model loaders  (cached to avoid re-reading on every interaction)
# ============================================================================


@st.cache_data(show_spinner=False)
def load_waste_reception() -> pd.DataFrame | None:
    p = DATA_RAW_DIR / "waste_reception_log.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_lab_analysis() -> pd.DataFrame | None:
    p = DATA_RAW_DIR / "laboratory_analysis.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_processing_batch() -> pd.DataFrame | None:
    p = DATA_RAW_DIR / "processing_batch.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # Try to parse timestamp columns if present
    for col in ["start_timestamp", "end_timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_separation_features() -> pd.DataFrame | None:
    p = DATA_FEATURES_DIR / "separation_features.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # Ensure processing_date is datetime; fall back to start_timestamp or similar
    if "processing_date" not in df.columns:
        for candidate in ["start_timestamp", "timestamp", "batch_date"]:
            if candidate in df.columns:
                df["processing_date"] = pd.to_datetime(df[candidate], errors="coerce")
                break
    else:
        df["processing_date"] = pd.to_datetime(df["processing_date"], errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def load_yield_model():
    """Load the trained XGBoost yield model artifact.

    joblib.load is safe here: the file is written exclusively by
    models/separation/yield_predictor.train_and_save() into the controlled
    local models/trained/ directory.  It is never sourced from user input or
    the network.
    """
    import joblib
    p = MODELS_DIR / "yield_model.joblib"
    if not p.exists():
        return None
    artifact = joblib.load(p)   # safe: project-local artifact only
    return artifact  # dict: {"model": model, "feature_columns": [...]}


@st.cache_resource(show_spinner=False)
def load_quality_classifier():
    """Load the trained quality pass/fail classifier artifact.

    joblib.load is safe here: the file is written exclusively by
    models/separation/quality_classifier.train_and_save() into the controlled
    local models/trained/ directory.  It is never sourced from user input or
    the network.
    """
    import joblib
    p = MODELS_DIR / "quality_classifier.joblib"
    if not p.exists():
        return None
    artifact = joblib.load(p)   # safe: project-local artifact only
    return artifact  # dict: {"pipeline": pipeline, "feature_columns": [...]}


@st.cache_data(show_spinner=False)
def load_yield_metrics() -> dict | None:
    p = MODELS_DIR / "yield_evaluation_metrics.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_quality_report() -> dict | None:
    p = MODELS_DIR / "classification_report.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_yield_predictions() -> pd.DataFrame | None:
    p = MODELS_DIR / "yield_predictions_vs_actuals.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


# ── helper: column description tooltip ──────────────────────────────────────

def _col_desc(col: str) -> str:
    return SEPARATION_COLUMN_DESCRIPTIONS.get(col, "")


def _safe_normalize(series: pd.Series) -> pd.Series:
    """Normalise a Series to 0-100; returns series of zeros if constant."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([50.0] * len(series), index=series.index)
    return (series - mn) / (mx - mn) * 100.0


# ============================================================================
# PAGE HEADER
# ============================================================================

st.title("Use Case 1: AI-Optimized Separation Process Control")
st.markdown(
    "ML-driven optimization of 3-phase petroleum waste separation "
    "to maximise oil recovery yield"
)
st.divider()

# ── pipeline data flow legend ────────────────────────────────────────────────
render_data_flow_explanation(
    source_tables=[
        ("waste_reception_log", 12_000, "MARPOL waste intake records per vessel call"),
        ("laboratory_analysis", 36_000, "Lab measurements per sample point (inlet/outlet)"),
        ("processing_batch", 12_000, "Centrifuge run results: yield, cost, quality verdict"),
    ],
    features_added=[
        ("viscosity_temperature_ratio", "viscosity_40c_cst / feed_temperature_c"),
        ("emulsion_difficulty_score", "water_pct × (1 + sulfur_pct) × ln(viscosity)"),
        ("contamination_index", "(V + Ni + Fe + Na) / 1000"),
        ("specific_energy_input", "energy_kwh / volume_m3"),
        ("capacity_utilization", "feed_flow_rate / max_capacity"),
        ("rolling_yield_7d", "7-day rolling mean yield at same facility"),
    ],
    model_name="XGBoost Regressor — predicts `oil_recovery_yield_pct`  \n"
               "Random Forest Classifier — predicts `quality_pass`",
    result_description="Scipy L-BFGS-B optimizer finds temperature, centrifuge speed, "
                       "flow rate & chemical dose that maximise predicted yield, "
                       "subject to flash-point safety constraint.",
)

st.divider()

# ============================================================================
# 5-TAB LAYOUT
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1 · Source Data",
    "📈 2 · Analytics",
    "⚙️ 3 · Feature Engineering",
    "🤖 4 · ML Model",
    "🎯 5 · Optimization Engine",
])


# ============================================================================
# TAB 1 — SOURCE DATA EXPLORER
# ============================================================================
with tab1:
    render_pipeline_steps(current_step=0)
    st.subheader("Source Data Explorer")
    st.markdown(
        "Three operational tables feed the separation ML pipeline. "
        "Select a table below to browse its contents."
    )

    wrl_df = load_waste_reception()
    lab_df = load_lab_analysis()
    pb_df  = load_processing_batch()

    missing_raw = [
        name for name, df in [
            ("waste_reception_log", wrl_df),
            ("laboratory_analysis", lab_df),
            ("processing_batch", pb_df),
        ] if df is None
    ]
    if missing_raw:
        st.error(
            f"Missing raw data files: {missing_raw}  \n"
            "Run `python scripts/generate_data.py` to create them."
        )
        st.stop()

    # ── financial KPI metrics ────────────────────────────────────────────────
    st.markdown("#### Financial Overview")
    fk1, fk2, fk3, fk4 = st.columns(4)

    total_revenue = 0.0
    avg_margin = 0.0
    total_volume = 0.0
    avg_acceptance_fee = 0.0

    if pb_df is not None and "recovered_oil_revenue_eur" in pb_df.columns:
        rev_cols = [c for c in ["recovered_oil_revenue_eur", "solid_fuel_revenue_eur"] if c in pb_df.columns]
        total_revenue = pb_df[rev_cols].sum().sum()

    if pb_df is not None and "net_margin_eur" in pb_df.columns:
        avg_margin = pb_df["net_margin_eur"].mean()

    if wrl_df is not None and "actual_volume_m3" in wrl_df.columns:
        total_volume = wrl_df["actual_volume_m3"].sum()

    if wrl_df is not None and "acceptance_fee_eur" in wrl_df.columns:
        avg_acceptance_fee = wrl_df["acceptance_fee_eur"].mean()
    elif pb_df is not None and "total_processing_cost_eur" in pb_df.columns:
        # Approximate acceptance fee from processing costs
        avg_acceptance_fee = pb_df["total_processing_cost_eur"].mean() * 0.15

    fk1.metric("Total Revenue", f"€{total_revenue:,.0f}")
    fk2.metric("Avg Margin / Batch", f"€{avg_margin:,.0f}")
    fk3.metric("Total Volume Processed", f"{total_volume:,.0f} m³")
    fk4.metric("Avg Acceptance Fee", f"€{avg_acceptance_fee:,.0f}")

    st.divider()

    # ── summary KPIs ────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Waste Reception Records", f"{len(wrl_df):,}")
    k2.metric("Lab Sample Records", f"{len(lab_df):,}")
    k3.metric("Processing Batches", f"{len(pb_df):,}")
    k4.metric(
        "Facilities",
        ", ".join(sorted(wrl_df["facility_code"].unique()))
        if "facility_code" in wrl_df.columns else "—",
    )

    if "timestamp" in wrl_df.columns:
        date_min = wrl_df["timestamp"].min().strftime("%d %b %Y")
        date_max = wrl_df["timestamp"].max().strftime("%d %b %Y")
        st.caption(f"Date range: **{date_min}** → **{date_max}**")

    st.divider()

    # ── table selector ───────────────────────────────────────────────────────
    table_choice = st.radio(
        "Browse table:",
        ["Waste Reception Log", "Laboratory Analysis", "Processing Batch"],
        horizontal=True,
    )

    TABLE_MAP = {
        "Waste Reception Log": wrl_df,
        "Laboratory Analysis": lab_df,
        "Processing Batch": pb_df,
    }
    sel_df = TABLE_MAP[table_choice]

    known_descs = {c: _col_desc(c) for c in sel_df.columns if _col_desc(c)}
    if known_descs:
        with st.expander("Column descriptions (hover/click for context)", expanded=False):
            for col, desc in known_descs.items():
                st.markdown(f"- **`{col}`** — {desc}")

    st.dataframe(sel_df.head(500), use_container_width=True)
    st.caption(f"{len(sel_df):,} rows × {len(sel_df.columns)} columns  (showing first 500)")

    st.divider()

    # ── charts ───────────────────────────────────────────────────────────────
    st.subheader("Visual Summaries")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Waste Volume by Facility (m³)**")
        if "actual_volume_m3" in wrl_df.columns and "facility_code" in wrl_df.columns:
            fig = waste_volume_by_facility_chart(wrl_df, colors=COLORS["facilities"])
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column `actual_volume_m3` not found in reception log.")

    with c2:
        st.markdown("**Waste Subcategory Distribution (volume)**")
        if "waste_subcategory" in wrl_df.columns and "actual_volume_m3" in wrl_df.columns:
            fig = waste_category_pie_chart(wrl_df)
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column `waste_subcategory` not found in reception log.")

    # Reception timeline
    if "timestamp" in wrl_df.columns and "actual_volume_m3" in wrl_df.columns:
        st.markdown("**Monthly Reception Volume by Facility**")
        ts = wrl_df.copy()
        ts["month"] = ts["timestamp"].dt.to_period("M").astype(str)
        monthly_vol = (
            ts.groupby(["month", "facility_code"])["actual_volume_m3"]
            .sum()
            .reset_index()
        )
        fig_ts = px.bar(
            monthly_vol,
            x="month",
            y="actual_volume_m3",
            color="facility_code",
            color_discrete_map=COLORS["facilities"],
            labels={
                "month": "Month",
                "actual_volume_m3": "Volume (m³)",
                "facility_code": "Facility",
            },
        )
        fig_ts.update_layout(height=320, margin=dict(t=20, b=40))
        apply_plotly_theme(fig_ts)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── cost breakdown donut ────────────────────────────────────────────────
    if pb_df is not None:
        cost_labels, cost_values = [], []
        for col, label in [
            ("chemicals_cost_eur", "Chemicals"),
            ("energy_consumed_kwh", "Energy"),
            ("labor_hours", "Labor"),
        ]:
            if col in pb_df.columns:
                cost_labels.append(label)
                cost_values.append(float(pb_df[col].sum()))

        if cost_labels:
            st.markdown("**Processing Cost Structure (Total)**")
            fig_donut = cost_breakdown_donut(
                labels=cost_labels,
                values=cost_values,
                title="Processing Cost Breakdown",
            )
            apply_plotly_theme(fig_donut)
            st.plotly_chart(fig_donut, use_container_width=True)


# ============================================================================
# TAB 2 — ANALYTICS  (NEW)
# ============================================================================
with tab2:
    render_pipeline_steps(current_step=1, total_steps=5)
    st.subheader("Operational Analytics")
    st.markdown(
        "Descriptive analysis of separation performance across facilities, "
        "operators, and waste types."
    )

    sep_features = load_separation_features()
    pb_df_t2 = load_processing_batch()

    if sep_features is None:
        st.info(
            "Feature file not found: `data/features/separation_features.csv`  \n"
            "Run `python scripts/engineer_features.py` to generate it."
        )
    else:
        # ── global facility filter ────────────────────────────────────────────
        available_facilities = sorted(sep_features["facility_code"].dropna().unique().tolist()) \
            if "facility_code" in sep_features.columns else []
        facility_filter = st.multiselect(
            "Filter by Facility",
            options=available_facilities,
            default=[],
            key="sep_analytics_facility",
        )
        filtered = (
            sep_features
            if not facility_filter
            else sep_features[sep_features["facility_code"].isin(facility_filter)]
        )
        st.caption(f"Showing **{len(filtered):,}** of {len(sep_features):,} batches")

        st.divider()

        # ── 1. Yield Distribution by Facility ────────────────────────────────
        if "oil_recovery_yield_pct" in filtered.columns and "facility_code" in filtered.columns:
            st.markdown("#### Yield Distribution by Facility")
            fig = violin_by_group(
                filtered,
                "oil_recovery_yield_pct",
                "facility_code",
                "Oil Recovery Yield Distribution by Facility",
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        # ── 2. Yield Trend Over Time ──────────────────────────────────────────
        if (
            "processing_date" in filtered.columns
            and "oil_recovery_yield_pct" in filtered.columns
            and filtered["processing_date"].notna().any()
        ):
            st.markdown("#### Yield Trend Over Time")
            tmp_ts = filtered.copy()
            tmp_ts["processing_date"] = pd.to_datetime(tmp_ts["processing_date"])
            if "facility_code" in tmp_ts.columns:
                monthly = (
                    tmp_ts.groupby(
                        [pd.Grouper(key="processing_date", freq="ME"), "facility_code"]
                    )["oil_recovery_yield_pct"]
                    .mean()
                    .reset_index()
                )
                color_col_ts = "facility_code"
            else:
                monthly = (
                    tmp_ts.groupby(pd.Grouper(key="processing_date", freq="ME"))[
                        "oil_recovery_yield_pct"
                    ]
                    .mean()
                    .reset_index()
                )
                color_col_ts = None
            fig = time_series_with_trend(
                monthly,
                "processing_date",
                "oil_recovery_yield_pct",
                color_col=color_col_ts,
                title="Monthly Average Oil Recovery Yield",
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        # ── 3. Facility Comparison Radar ──────────────────────────────────────
        if "facility_code" in filtered.columns:
            st.markdown("#### Facility Performance Comparison")
            facilities = sorted(filtered["facility_code"].dropna().unique().tolist())
            if len(facilities) >= 2:
                categories = [
                    "Avg Yield",
                    "Quality Pass %",
                    "Avg Margin (€K)",
                    "Volume (K m³)",
                    "Energy Efficiency",
                    "Equipment Condition",
                ]
                values_dict: dict = {}
                raw_vals: dict = {fac: [] for fac in facilities}
                for fac in facilities:
                    grp = filtered[filtered["facility_code"] == fac]
                    # Avg Yield
                    avg_yld = grp["oil_recovery_yield_pct"].mean() if "oil_recovery_yield_pct" in grp.columns else 0.0
                    # Quality Pass %
                    if "quality_pass" in grp.columns:
                        q_pass = grp["quality_pass"].mean() * 100.0
                    else:
                        q_pass = avg_yld  # proxy
                    # Avg Margin
                    avg_margin_fac = grp["net_margin_eur"].mean() / 1000.0 if "net_margin_eur" in grp.columns else 0.0
                    # Volume — try to get from reception or features
                    if pb_df_t2 is not None and "facility_code" in pb_df_t2.columns and "total_processing_cost_eur" in pb_df_t2.columns:
                        vol_fac = len(pb_df_t2[pb_df_t2["facility_code"] == fac]) / 1000.0
                    else:
                        vol_fac = len(grp) / 1000.0
                    # Energy efficiency (inverse of specific energy input — lower is better)
                    if "specific_energy_input" in grp.columns:
                        sei = grp["specific_energy_input"].mean()
                        energy_eff = 1.0 / sei if sei > 0 else 0.0
                    else:
                        energy_eff = avg_yld  # proxy
                    # Equipment condition
                    eq_cond = grp["equipment_condition_score"].mean() if "equipment_condition_score" in grp.columns else 50.0

                    raw_vals[fac] = [avg_yld, q_pass, avg_margin_fac, vol_fac, energy_eff, eq_cond]

                # Normalise each category to 0-100 across facilities
                n_cats = len(categories)
                norm_matrix: dict = {fac: [0.0] * n_cats for fac in facilities}
                for ci in range(n_cats):
                    col_vals = [raw_vals[fac][ci] for fac in facilities]
                    mn_v, mx_v = min(col_vals), max(col_vals)
                    for fac in facilities:
                        if mx_v == mn_v:
                            norm_matrix[fac][ci] = 50.0
                        else:
                            norm_matrix[fac][ci] = (raw_vals[fac][ci] - mn_v) / (mx_v - mn_v) * 100.0

                for fac in facilities:
                    values_dict[fac] = norm_matrix[fac]

                fig = radar_chart(categories, values_dict, "Facility KPI Comparison (normalised 0-100)")
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least 2 facilities (or clear the filter) to see the radar chart.")

        # ── 4. Waste Category Profitability ───────────────────────────────────
        if "waste_subcategory" in filtered.columns and "net_margin_eur" in filtered.columns:
            st.markdown("#### Profitability by Waste Category")
            waste_profit = (
                filtered.groupby("waste_subcategory")["net_margin_eur"]
                .mean()
                .sort_values(ascending=True)
                .reset_index()
            )
            fig = px.bar(
                waste_profit,
                x="net_margin_eur",
                y="waste_subcategory",
                orientation="h",
                title="Average Net Margin by Waste Type (€)",
                labels={"net_margin_eur": "Avg Net Margin (€)", "waste_subcategory": "Waste Type"},
                color="net_margin_eur",
                color_continuous_scale=["#E53935", "#FFFFFF", "#1B9AAA"],
                color_continuous_midpoint=0,
            )
            fig.update_layout(height=420, coloraxis_showscale=False)
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        # ── 5. Operator Performance ───────────────────────────────────────────
        if "operator_id" in filtered.columns and "oil_recovery_yield_pct" in filtered.columns:
            st.markdown("#### Operator Performance")
            top_ops = filtered["operator_id"].value_counts().head(15).index
            op_data = filtered[filtered["operator_id"].isin(top_ops)].copy()
            op_data["operator_id"] = op_data["operator_id"].astype(str)
            fig = px.box(
                op_data,
                x="operator_id",
                y="oil_recovery_yield_pct",
                color="facility_code" if "facility_code" in op_data.columns else None,
                color_discrete_map=COLORS.get("facilities"),
                title="Yield by Operator (top 15 by batch count)",
                labels={
                    "operator_id": "Operator ID",
                    "oil_recovery_yield_pct": "Oil Recovery Yield (%)",
                },
            )
            fig.update_layout(height=420)
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        # ── 6. Equipment Condition vs Yield ───────────────────────────────────
        if "equipment_condition_score" in filtered.columns and "oil_recovery_yield_pct" in filtered.columns:
            st.markdown("#### Equipment Degradation Analysis")
            fig = scatter_with_trend(
                filtered.sample(min(3000, len(filtered)), random_state=42),
                "equipment_condition_score",
                "oil_recovery_yield_pct",
                color_col="facility_code" if "facility_code" in filtered.columns else None,
                title="Equipment Condition Score vs Oil Recovery Yield",
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        # ── 7. Cost Breakdown ─────────────────────────────────────────────────
        st.markdown("#### Processing Cost Structure")
        if pb_df_t2 is not None:
            cost_labels_t2, cost_values_t2 = [], []
            for col, label in [
                ("chemicals_cost_eur", "Chemicals"),
                ("energy_consumed_kwh", "Energy (kWh equiv.)"),
                ("labor_hours", "Labor (hrs equiv.)"),
            ]:
                if col in pb_df_t2.columns:
                    cost_labels_t2.append(label)
                    cost_values_t2.append(float(pb_df_t2[col].sum()))

            if cost_labels_t2:
                fig = cost_breakdown_donut(
                    labels=cost_labels_t2,
                    values=cost_values_t2,
                    title="Total Processing Cost Breakdown",
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Cost columns not found in `processing_batch.csv`.")
        else:
            st.info("Processing batch data not available for cost breakdown.")

        # ── 8. Profitability Distribution ─────────────────────────────────────
        if "net_margin_eur" in filtered.columns:
            st.markdown("#### Margin Distribution by Facility")
            fig = profitability_distribution(
                filtered,
                margin_col="net_margin_eur",
                group_col="facility_code" if "facility_code" in filtered.columns else None,
                title="Net Margin Distribution per Batch (€)",
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        # ── 9. Correlation Explorer ───────────────────────────────────────────
        st.markdown("#### Feature Correlation Explorer")
        numeric_cols_t2 = [
            c for c in filtered.select_dtypes(include=[np.number]).columns
            if c not in {"batch_id", "quality_pass"}
        ]
        default_corr_cols = [
            c for c in [
                "oil_recovery_yield_pct",
                "viscosity_40c_cst",
                "water_content_pct",
                "emulsion_difficulty_score",
                "equipment_condition_score",
                "contamination_index",
            ]
            if c in numeric_cols_t2
        ]
        corr_cols = st.multiselect(
            "Select features to correlate",
            options=numeric_cols_t2,
            default=default_corr_cols or numeric_cols_t2[:6],
            key="sep_analytics_corr",
        )
        if len(corr_cols) >= 2:
            corr_sample = filtered[corr_cols].dropna().sample(
                min(2000, len(filtered)), random_state=42
            )
            fig_scatter = px.scatter_matrix(
                corr_sample,
                dimensions=corr_cols,
                color="facility_code" if "facility_code" in filtered.columns else None,
                color_discrete_map=COLORS.get("facilities"),
                title="Feature Scatter Matrix",
                opacity=0.4,
            )
            fig_scatter.update_traces(marker=dict(size=3))
            fig_scatter.update_layout(height=600)
            apply_plotly_theme(fig_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Select at least 2 features to display the scatter matrix.")


# ============================================================================
# TAB 3 — FEATURE ENGINEERING PIPELINE
# ============================================================================
with tab3:
    render_pipeline_steps(current_step=2, total_steps=5)
    st.subheader("Feature Engineering Pipeline")
    st.markdown(
        "The raw tables are joined, cleaned, and enriched to produce the "
        "`separation_features.csv` feature matrix (12,178 rows × 48 columns). "
        "Columns highlighted in **green** are engineered features; others come "
        "directly from the source tables."
    )

    feat_df = load_separation_features()
    if feat_df is None:
        st.error(
            "Feature file not found: `data/features/separation_features.csv`  \n"
            "Run `python scripts/engineer_features.py` to generate it."
        )
        st.stop()

    # ── feature matrix overview ──────────────────────────────────────────────
    ENGINEERED_COLS = {
        "viscosity_temperature_ratio",
        "emulsion_difficulty_score",
        "oil_water_density_gap",
        "solid_particle_settling_velocity",
        "waste_age_degradation",
        "sulfur_to_oil_ratio",
        "contamination_index",
        "cat_fines_risk",
        "specific_energy_input",
        "g_force",
        "chemical_to_emulsion_ratio",
        "temperature_vs_pour_point_margin",
        "capacity_utilization",
        "hour_of_day",
        "day_of_week",
        "month",
        "is_night_shift",
        "is_weekend",
        "rolling_yield_7d",
        "rolling_energy_7d",
        "equipment_hours_since_service",
        "similar_batch_avg_yield",
        "operator_avg_yield",
        "ambient_heating_delta",
        "viscosity_x_flow_rate",
        "solids_x_rpm",
        "water_x_temperature",
        "sulfur_x_volume",
        "age_x_emulsion",
        "residence_time_actual",
    }

    all_cols = feat_df.columns.tolist()
    raw_cols = [c for c in all_cols if c not in ENGINEERED_COLS]
    eng_cols = [c for c in all_cols if c in ENGINEERED_COLS]

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Feature Columns", len(all_cols))
    k2.metric("Raw / Pass-through Columns", len(raw_cols))
    k3.metric("Engineered Features", len(eng_cols))

    st.divider()

    # ── formula explanations ─────────────────────────────────────────────────
    st.subheader("Key Engineered Features & Formulas")

    FORMULAS = [
        ("viscosity_temperature_ratio",
         "`viscosity_40c_cst / feed_temperature_c`  \n"
         "Normalised pumpability — higher ratio means harder to pump at that temperature."),
        ("emulsion_difficulty_score",
         "`water_content_pct × (1 + sulfur_total_pct) × ln(viscosity_40c_cst)`  \n"
         "Composite difficulty of breaking the oil-water emulsion."),
        ("contamination_index",
         "`(vanadium_ppm + nickel_ppm + iron_ppm + sodium_ppm) / 1000`  \n"
         "Metal contamination severity; high values reduce catalyst activity."),
        ("specific_energy_input",
         "`energy_consumed_kwh / total_volume_input_m3`  \n"
         "Energy intensity per unit volume — proxy for process efficiency."),
        ("capacity_utilization",
         "`feed_flow_rate_m3_hr / rated_max_capacity_m3_hr`  \n"
         "Equipment loading fraction; over-loading degrades separation."),
        ("rolling_yield_7d",
         "`rolling(7d, facility).mean(oil_recovery_yield_pct)`  \n"
         "7-day rolling average yield at same facility — encodes recent equipment/operational state."),
        ("g_force",
         "`(centrifuge_speed_rpm² × bowl_radius_m) / (900 × g)`  \n"
         "Centrifugal force multiple — key driver of solid/liquid separation efficiency."),
    ]

    for feat, desc in FORMULAS:
        if feat in eng_cols:
            with st.expander(f"⚙️ `{feat}`", expanded=False):
                st.markdown(desc)
                stats = feat_df[feat].describe()
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Mean", f"{stats['mean']:.3g}")
                col_b.metric("Std", f"{stats['std']:.3g}")
                col_c.metric("Min", f"{stats['min']:.3g}")
                col_d.metric("Max", f"{stats['max']:.3g}")

    st.divider()

    # ── feature browser ──────────────────────────────────────────────────────
    st.subheader("Browse Feature Matrix")
    show_cols = st.multiselect(
        "Columns to display",
        options=all_cols,
        default=[
            c for c in [
                "batch_id",
                "oil_recovery_yield_pct",
                "viscosity_40c_cst",
                "water_content_pct",
                "emulsion_difficulty_score",
                "contamination_index",
                "rolling_yield_7d",
                "capacity_utilization",
                "quality_pass",
            ]
            if c in all_cols
        ],
    )
    if show_cols:
        st.dataframe(feat_df[show_cols].head(300), use_container_width=True)

    st.divider()

    # ── correlation heatmap ──────────────────────────────────────────────────
    st.subheader("Feature Correlations with Target (oil_recovery_yield_pct)")
    top_n_corr = st.slider("Number of top features", 10, 30, 18, key="corr_n")
    if "oil_recovery_yield_pct" in feat_df.columns:
        fig_hm = correlation_heatmap(feat_df, "oil_recovery_yield_pct", top_n=top_n_corr)
        apply_plotly_theme(fig_hm)
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Target column `oil_recovery_yield_pct` not found in feature matrix.")

    st.divider()

    # ── feature distributions ────────────────────────────────────────────────
    st.subheader("Feature Distributions")
    numeric_feat_cols = [
        c for c in feat_df.select_dtypes(include=[np.number]).columns
        if c not in {"batch_id", "quality_pass"}
    ]
    dist_col = st.selectbox(
        "Select feature",
        options=numeric_feat_cols,
        index=numeric_feat_cols.index("oil_recovery_yield_pct")
        if "oil_recovery_yield_pct" in numeric_feat_cols else 0,
    )

    fig_hist = px.histogram(
        feat_df,
        x=dist_col,
        nbins=60,
        color_discrete_sequence=[COLORS["primary"]],
        labels={dist_col: dist_col},
        opacity=0.8,
    )
    fig_hist.update_layout(height=300, margin=dict(t=20, b=30))
    apply_plotly_theme(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True)


# ============================================================================
# TAB 4 — ML MODEL PERFORMANCE
# ============================================================================
with tab4:
    render_pipeline_steps(current_step=3, total_steps=5)
    st.subheader("ML Model Performance")

    yield_artifact = load_yield_model()
    quality_artifact = load_quality_classifier()
    yield_metrics = load_yield_metrics()
    quality_report = load_quality_report()
    preds_df = load_yield_predictions()

    if yield_artifact is None:
        st.error(
            "Yield model not found: `models/trained/yield_model.joblib`  \n"
            "Run `python scripts/train_models.py` to train the models."
        )
        st.stop()

    model = yield_artifact["model"]
    feat_cols = yield_artifact.get("feature_columns", yield_artifact.get("feature_cols", []))

    # ── yield regressor section ──────────────────────────────────────────────
    st.markdown("### XGBoost Yield Regressor — `oil_recovery_yield_pct`")
    st.markdown(
        "Trained on 80% of batches (time-based split). "
        "Predicts the percentage of oil recovered from the feedstock."
    )

    if yield_metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("R²", f"{yield_metrics['r2']:.3f}")
        m2.metric("RMSE (pp)", f"{yield_metrics['rmse']:.2f}")
        m3.metric("MAE (pp)", f"{yield_metrics['mae']:.2f}")
        m4.metric("Train Batches", f"{yield_metrics['train_size']:,}")
        m5.metric("Test Batches", f"{yield_metrics['test_size']:,}")
    else:
        st.info("Metrics file not found — showing model only.")

    st.divider()

    c_imp, c_pred = st.columns([1, 1])

    with c_imp:
        st.markdown("**Top-15 Feature Importances**")
        fig_fi = feature_importance_chart(model, feat_cols, top_n=15)
        apply_plotly_theme(fig_fi)
        st.plotly_chart(fig_fi, use_container_width=True)

    with c_pred:
        st.markdown("**Actual vs Predicted Yield (%)**")
        if preds_df is not None:
            fig_avp = actual_vs_predicted_chart(
                preds_df["actual"],
                preds_df["predicted"].values,
            )
            apply_plotly_theme(fig_avp)
            st.plotly_chart(fig_avp, use_container_width=True)
        else:
            st.info("Predictions file not available.")

    # Residuals
    if preds_df is not None:
        st.markdown("**Residual Distribution (actual − predicted)**")
        fig_res = residual_distribution_chart(preds_df["residual"].values)
        apply_plotly_theme(fig_res)
        st.plotly_chart(fig_res, use_container_width=True)

    st.divider()

    # ── quality classifier section ───────────────────────────────────────────
    st.markdown("### Random Forest Quality Classifier — `quality_pass`")
    st.markdown(
        "Predicts whether a batch will meet buyer quality specifications (pass/fail). "
        "Same 80/20 time-based split."
    )

    if quality_artifact is None:
        st.warning("Quality classifier not loaded — skipping this section.")
    else:
        if quality_report:
            qm1, qm2, qm3, qm4 = st.columns(4)
            qm1.metric("Accuracy", f"{quality_report.get('accuracy', 0):.3f}")
            qm2.metric("Precision", f"{quality_report.get('precision', 0):.3f}")
            qm3.metric("Recall", f"{quality_report.get('recall', 0):.3f}")
            qm4.metric("F1 Score", f"{quality_report.get('f1', 0):.3f}")

            cm = quality_report.get("confusion_matrix")
            if cm:
                st.markdown("**Confusion Matrix**")
                fig_cm = confusion_matrix_chart(cm, labels=["Fail", "Pass"])
                apply_plotly_theme(fig_cm)
                st.plotly_chart(fig_cm, use_container_width=False)
        else:
            st.info("Classification report not available.")

        rf = quality_artifact["pipeline"].named_steps["clf"]
        q_feat_cols = quality_artifact.get("feature_columns", quality_artifact.get("feature_cols", []))
        st.markdown("**Quality Classifier — Feature Importances (top 15)**")
        fig_qfi = feature_importance_chart(rf, q_feat_cols, top_n=15)
        apply_plotly_theme(fig_qfi)
        st.plotly_chart(fig_qfi, use_container_width=True)


# ============================================================================
# TAB 5 — OPTIMIZATION ENGINE
# ============================================================================
with tab5:
    render_pipeline_steps(current_step=4, total_steps=5)
    st.subheader("Optimization Engine")
    st.markdown(
        "The optimizer uses the trained XGBoost model as a surrogate objective and "
        "runs **scipy L-BFGS-B** to find process parameters that maximise predicted "
        "oil recovery yield — subject to safety and operational constraints."
    )

    # ── financial context callout ─────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:#FFF8E1; border-left:5px solid #F9A825; border-radius:8px;
                    padding:14px 20px; margin:12px 0; font-size:0.9rem; color:#546E7A;">
            <strong>Financial Scale:</strong>
            Each <strong>1% yield improvement</strong> = <strong>€1,500 / batch</strong>
            = <strong>€18M / year</strong> across all HEC facilities (12,000 batches/year).
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        from models.separation.parameter_optimizer import (
            optimize_parameters,
            what_if_analysis,
            TUNABLE_PARAMS,
            BASELINE_PARAMS,
        )
        optimizer_available = True
    except Exception as exc:
        st.error(f"Could not import parameter optimizer: {exc}")
        optimizer_available = False

    if not optimizer_available:
        st.stop()

    yield_artifact_5 = load_yield_model()
    feat_df_5 = load_separation_features()

    if yield_artifact_5 is None:
        st.error("Yield model not found. Run `python scripts/train_models.py` first.")
        st.stop()

    if feat_df_5 is None:
        st.error(
            "Feature matrix not found. Run `python scripts/engineer_features.py` first."
        )
        st.stop()

    st.divider()

    # ── batch selector ───────────────────────────────────────────────────────
    st.markdown("### Step 1 — Select or enter feedstock characteristics")
    input_mode = st.radio(
        "Input mode",
        ["Pick a sample batch from the dataset", "Enter custom values"],
        horizontal=True,
    )

    FIXED_FEEDSTOCK_COLS = [
        "viscosity_40c_cst",
        "water_content_pct",
        "oil_content_pct",
        "solids_content_pct",
        "density_15c_kg_m3",
        "flash_point_c",
        "sulfur_total_pct",
        "emulsion_layer_pct",
        "contamination_index",
        "particle_size_d50_micron",
        "pour_point_c",
    ]
    FIXED_FEEDSTOCK_COLS = [c for c in FIXED_FEEDSTOCK_COLS if c in feat_df_5.columns]

    batch_chars: dict = {}

    if input_mode == "Pick a sample batch from the dataset":
        sample_size = min(200, len(feat_df_5))
        sample_indices = feat_df_5.index[:sample_size].tolist()
        row_idx = st.selectbox(
            "Batch index (first 200 batches shown)",
            options=sample_indices,
            format_func=lambda i: f"Batch #{i} — yield={feat_df_5.loc[i,'oil_recovery_yield_pct']:.1f}%"
            if "oil_recovery_yield_pct" in feat_df_5.columns else f"Batch #{i}",
        )
        selected_row = feat_df_5.loc[row_idx]
        batch_chars = {
            c: float(selected_row[c])
            for c in FIXED_FEEDSTOCK_COLS
            if pd.notna(selected_row.get(c, np.nan))
        }

        with st.expander("Selected batch properties", expanded=True):
            prop_cols = st.columns(4)
            for j, (k, v) in enumerate(batch_chars.items()):
                prop_cols[j % 4].metric(k, f"{v:.3g}")
    else:
        st.markdown("Enter feedstock properties (leave blank to use dataset mean):")
        feat_means = feat_df_5[FIXED_FEEDSTOCK_COLS].mean()

        c1_t5, c2_t5, c3_t5 = st.columns(3)
        for j, col_name in enumerate(FIXED_FEEDSTOCK_COLS):
            widget_col = [c1_t5, c2_t5, c3_t5][j % 3]
            default_val = float(feat_means[col_name]) if col_name in feat_means else 0.0
            val = widget_col.number_input(
                col_name,
                value=round(default_val, 2),
                key=f"custom_{col_name}",
            )
            batch_chars[col_name] = val

    st.divider()

    # ── safety constraint: flash point ──────────────────────────────────────
    flash_pt = batch_chars.get("flash_point_c")
    if flash_pt is not None:
        max_temp_safe = flash_pt - 10.0
        st.info(
            f"Safety constraint: max process temperature = flash point − 10°C "
            f"= **{max_temp_safe:.1f}°C**"
        )
    else:
        max_temp_safe = 95.0

    # ── optimize button ──────────────────────────────────────────────────────
    st.markdown("### Step 2 — Run optimizer")

    if st.button("▶ Optimize Parameters", type="primary"):
        with st.spinner("Running scipy L-BFGS-B optimization…"):
            try:
                result = optimize_parameters(
                    batch_characteristics=batch_chars,
                    constraints={
                        "max_temperature_c": max_temp_safe,
                        "max_capacity_utilization": 0.95,
                    },
                )
                st.session_state["opt_result"] = result
                st.session_state["opt_batch_chars"] = batch_chars.copy()
            except Exception as exc:
                st.error(f"Optimization failed: {exc}")
                st.session_state.pop("opt_result", None)

    # ── results ──────────────────────────────────────────────────────────────
    if "opt_result" in st.session_state:
        result = st.session_state["opt_result"]
        opt_params = result["optimal_parameters"]
        pred_yield = result["predicted_yield"]
        improvement = result["yield_improvement_vs_baseline"]
        rev_impact = result["estimated_revenue_impact_eur"]

        st.divider()
        st.markdown("### Step 3 — Results")

        # ── primary financial callouts ────────────────────────────────────────
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Yield (optimised)", f"{pred_yield:.2f}%")
        delta_str = f"{improvement:+.2f} pp"
        r2.metric(
            "Improvement vs Baseline",
            delta_str,
            delta=delta_str,
            delta_color="normal" if improvement >= 0 else "inverse",
        )
        r3.metric("Est. Revenue Impact / Batch", f"€{rev_impact:,.0f}")

        st.divider()

        # ── prominent financial callout ───────────────────────────────────────
        if improvement > 0:
            render_money_callout(
                title="Revenue Impact — This Batch",
                amount_eur=rev_impact,
                description=(
                    f"Optimised yield of {pred_yield:.2f}% vs baseline "
                    f"(+{improvement:.2f} percentage points)"
                ),
            )

            # Annual extrapolation
            annual_batches = 12_000
            revenue_per_pp = 1_500.0
            annual_impact = improvement * revenue_per_pp * annual_batches
            render_money_callout(
                title="Annual Extrapolation (all facilities)",
                amount_eur=annual_impact,
                description=(
                    f"{improvement:.2f} pp improvement × €{revenue_per_pp:,.0f}/batch/pp "
                    f"× {annual_batches:,} batches/year"
                ),
            )

            st.markdown(
                """
                <div style="background:#E8F5E9; border-left:4px solid #43A047;
                            border-radius:6px; padding:12px 18px; margin:8px 0;
                            font-size:0.88rem; color:#2E7D32;">
                    <strong>Sensitivity rule of thumb:</strong>
                    Each <strong>1% yield improvement</strong> =
                    <strong>€1,500 / batch</strong> =
                    <strong>€18,000,000 / year</strong> across all HEC facilities.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ── parameter comparison table ────────────────────────────────────────
        st.markdown("**Optimal Parameters vs Baseline**")
        param_names_disp = list(BASELINE_PARAMS.keys())
        baseline_vals = [BASELINE_PARAMS[p] for p in param_names_disp]
        optimised_vals = [opt_params.get(p, np.nan) for p in param_names_disp]

        comp_df = pd.DataFrame({
            "Parameter": param_names_disp,
            "Baseline": [round(v, 3) for v in baseline_vals],
            "Optimised": [round(v, 3) if not np.isnan(v) else "—" for v in optimised_vals],
        })
        comp_df["Change"] = comp_df.apply(
            lambda row: (
                f"{(row['Optimised'] - row['Baseline']):.3g}"
                if isinstance(row["Optimised"], (int, float)) else "—"
            ),
            axis=1,
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # ── before/after chart ───────────────────────────────────────────────
        valid_mask = [not np.isnan(v) for v in optimised_vals]
        chart_names  = [n for n, v in zip(param_names_disp, valid_mask) if v]
        chart_before = [b for b, v in zip(baseline_vals,   valid_mask) if v]
        chart_after  = [a for a, v in zip(optimised_vals,  valid_mask) if v]

        if chart_names:
            st.markdown("**Before / After Parameter Comparison**")
            fig_cmp = optimization_comparison_chart(chart_before, chart_after, chart_names)
            apply_plotly_theme(fig_cmp)
            st.plotly_chart(fig_cmp, use_container_width=True)

        # ── what-if section ──────────────────────────────────────────────────
        st.divider()
        st.markdown("### Interactive What-If Analysis")
        st.markdown(
            "Manually adjust individual parameters and immediately see the "
            "predicted yield change versus baseline."
        )

        wa_cols_list = st.columns(len(TUNABLE_PARAMS))
        wa_overrides: dict = {}
        for j, (param, (lo, hi)) in enumerate(TUNABLE_PARAMS.items()):
            with wa_cols_list[j % len(TUNABLE_PARAMS)]:
                default_val = float(BASELINE_PARAMS.get(param, (lo + hi) / 2))
                wa_overrides[param] = st.slider(
                    param,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=default_val,
                    key=f"wa_{param}",
                )

        wa_result = what_if_analysis(
            batch_characteristics=st.session_state.get("opt_batch_chars", batch_chars),
            parameter_overrides=wa_overrides,
        )

        wa1, wa2, wa3 = st.columns(3)
        wa1.metric("Predicted Yield", f"{wa_result['predicted_yield']:.2f}%")
        wa_delta = wa_result["delta_yield"]
        wa2.metric(
            "vs Baseline",
            f"{wa_delta:+.2f} pp",
            delta=f"{wa_delta:+.2f} pp",
            delta_color="normal" if wa_delta >= 0 else "inverse",
        )
        wa3.metric("Baseline Yield", f"{wa_result['baseline_yield']:.2f}%")

        # Financial impact of what-if selection
        wa_revenue = wa_delta * 1_500.0
        wa_annual  = wa_delta * 1_500.0 * 12_000
        st.markdown(
            f"**What-If Financial Impact:** "
            f"€{wa_revenue:,.0f} / batch &nbsp;|&nbsp; "
            f"€{wa_annual:,.0f} / year (extrapolated across all facilities)"
        )

    else:
        st.info("Click **▶ Optimize Parameters** to run the optimizer.")

    st.divider()

    # ── constraints legend ───────────────────────────────────────────────────
    with st.expander("Optimization constraints & parameter bounds", expanded=False):
        st.markdown(
            "| Parameter | Lower Bound | Upper Bound | Notes |\n"
            "|---|---|---|---|\n"
            + "\n".join(
                f"| `{p}` | {lo} | {hi} |"
                + (" Max capped at flash point − 10°C" if p == "feed_temperature_c" else "")
                + " |"
                for p, (lo, hi) in TUNABLE_PARAMS.items()
            )
        )
        st.markdown(
            "**Safety rule:** `feed_temperature_c` ≤ `flash_point_c − 10°C`  \n"
            "This ensures a 10°C safety margin below the ignition point of the feedstock."
        )
