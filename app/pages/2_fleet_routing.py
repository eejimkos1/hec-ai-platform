"""Use Case 2: Predictive Fleet Routing & Demand Forecasting.

Streamlit page with 5 tabs matching the ML pipeline:
  Tab 1 — Source Data Explorer (with financial KPIs)
  Tab 2 — Analytics (NEW — descriptive analysis)
  Tab 3 — Demand Forecast Pipeline (Feature Engineering)
  Tab 4 — ML Model Performance
  Tab 5 — Fleet Optimization Dashboard (Results)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `app.*` imports resolve correctly
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # …/meli/app/pages
_ROOT = _HERE.parent.parent.parent               # …/meli
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import (
    COLORS,
    DATA_FEATURES_DIR,
    DATA_RAW_DIR,
    DATA_REFERENCE_DIR,
    FLEET_COLUMN_DESCRIPTIONS,
    MODELS_DIR,
)
from app.components.theme import inject_hec_css, render_money_callout, apply_plotly_theme
from app.components.pipeline_viewer import render_data_flow_explanation, render_pipeline_steps
from app.components.map_viewer import render_port_map
from app.components.charts import (
    violin_by_group,
    time_series_with_trend,
    profitability_distribution,
    dual_axis_chart,
    scatter_with_trend,
    calendar_heatmap,
    cost_breakdown_donut,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HEC Fleet Routing — Use Case 2",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_hec_css()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛢️ HEC AI Platform")
    st.caption("Hellenic Environmental Center")
    st.divider()
    st.markdown("**Use Case 2** — Fleet Routing & Demand Forecasting")
    st.markdown(
        "Use the tabs above to walk through the full ML pipeline from raw data "
        "to optimized voyage schedules."
    )
    st.divider()
    st.info("Navigate to **Use Case 1** via the sidebar page list for Separation Control.")

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_ports() -> pd.DataFrame | None:
    p = DATA_REFERENCE_DIR / "ports.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_fleet_master() -> pd.DataFrame | None:
    p = DATA_REFERENCE_DIR / "fleet_master.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_distance_matrix() -> pd.DataFrame | None:
    p = DATA_REFERENCE_DIR / "port_distance_matrix.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_port_waste_demand() -> pd.DataFrame | None:
    p = DATA_RAW_DIR / "port_waste_demand.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_voyage_history() -> pd.DataFrame | None:
    p = DATA_RAW_DIR / "voyage_history.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["departure_time"] = pd.to_datetime(df["departure_time"])
    df["arrival_time"] = pd.to_datetime(df["arrival_time"])
    return df


@st.cache_data(show_spinner=False)
def load_fleet_status() -> pd.DataFrame | None:
    p = DATA_RAW_DIR / "fleet_status.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_oil_market() -> pd.DataFrame | None:
    p = DATA_RAW_DIR / "oil_market.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_maritime_weather() -> pd.DataFrame | None:
    p = DATA_RAW_DIR / "maritime_weather.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_fleet_features() -> pd.DataFrame | None:
    p = DATA_FEATURES_DIR / "fleet_features.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_route_features() -> pd.DataFrame | None:
    p = DATA_FEATURES_DIR / "route_features.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_resource(show_spinner=False)
def load_demand_model() -> dict | None:
    p = MODELS_DIR / "demand_forecaster.joblib"
    if not p.exists():
        return None
    import joblib
    # Safe: written exclusively by models/fleet/demand_forecaster.py (our own pipeline).
    return joblib.load(p)


@st.cache_data(show_spinner=False)
def load_demand_metrics() -> dict | None:
    p = MODELS_DIR / "demand_forecaster_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_demand_predictions() -> pd.DataFrame | None:
    p = MODELS_DIR / "demand_forecaster_predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_profitability_metrics() -> dict | None:
    p = MODELS_DIR / "profitability_model_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_profitability_feature_importances() -> pd.DataFrame | None:
    p = MODELS_DIR / "profitability_feature_importances.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_demand_port_metrics() -> pd.DataFrame | None:
    p = MODELS_DIR / "demand_forecaster_port_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Use Case 2: Predictive Fleet Routing & Demand Forecasting")
st.markdown(
    "ML demand forecasting + route optimization for HEC's 25-vessel collection fleet"
)
st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 1 · Source Data",
        "📈 2 · Analytics",
        "⚙️ 3 · Demand Forecast Pipeline",
        "🤖 4 · ML Model Performance",
        "🎯 5 · Fleet Optimization",
    ]
)

# ===========================================================================
# TAB 1 — SOURCE DATA EXPLORER
# ===========================================================================
with tab1:
    render_pipeline_steps(current_step=0)
    st.divider()

    st.subheader("Source Data Explorer")

    ports_df = load_ports()
    demand_df = load_port_waste_demand()
    voyage_df = load_voyage_history()
    fleet_status_df = load_fleet_status()
    oil_market_df = load_oil_market()

    # --- Operational stats row ---
    if ports_df is not None and demand_df is not None and voyage_df is not None:
        date_min = demand_df["date"].min().strftime("%Y-%m-%d")
        date_max = demand_df["date"].max().strftime("%Y-%m-%d")
        n_ports = ports_df["port_code"].nunique()
        n_vessels = (
            fleet_status_df["vessel_id"].nunique() if fleet_status_df is not None else 25
        )
        n_voyages = len(voyage_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Date Range", f"{date_min}  →  {date_max}")
        c2.metric("Ports Covered", n_ports)
        c3.metric("Vessels in Fleet", n_vessels)
        c4.metric("Total Voyages", f"{n_voyages:,}")
    else:
        st.warning("Run the pipeline scripts to generate raw data files.")

    # --- Financial KPIs from voyage history ---
    if voyage_df is not None:
        st.markdown("#### Fleet Financial Summary")
        total_revenue = voyage_df["revenue_eur"].sum() if "revenue_eur" in voyage_df.columns else 0.0
        total_fuel = voyage_df["fuel_cost_eur"].sum() if "fuel_cost_eur" in voyage_df.columns else 0.0
        total_margin = voyage_df["voyage_margin_eur"].sum() if "voyage_margin_eur" in voyage_df.columns else 0.0
        avg_margin = voyage_df["voyage_margin_eur"].mean() if "voyage_margin_eur" in voyage_df.columns else 0.0

        fk1, fk2, fk3, fk4 = st.columns(4)
        fk1.metric("Total Fleet Revenue", f"€{total_revenue:,.0f}")
        fk2.metric("Total Fuel Costs", f"€{total_fuel:,.0f}")
        fk3.metric("Net Fleet Margin", f"€{total_margin:,.0f}")
        fk4.metric("Avg Margin / Voyage", f"€{avg_margin:,.0f}")

    st.divider()

    # --- Interactive port map ---
    st.subheader("HEC Port Network — 14 Mediterranean & Northern European Ports")
    if ports_df is not None:
        latest_demand = None
        if demand_df is not None:
            latest_date = demand_df["date"].max()
            latest_demand = demand_df[demand_df["date"] == latest_date].copy()

        port_map, h = render_port_map(ports_df, demand_df=latest_demand, height=480)
        st_folium(port_map, height=h, width=None)
        st.caption(
            "Circle size = latest daily waste volume. Color: green < 50% storage, "
            "orange = 50–75%, red > 75% fill level."
        )
    else:
        st.warning("ports.csv not found. Run `python scripts/generate_data.py`.")

    st.divider()

    # --- Source table explorer ---
    st.subheader("Explore Source Tables")

    TABLE_OPTIONS = {
        "Port Waste Demand (port_waste_demand.csv)": demand_df,
        "Voyage History (voyage_history.csv)": voyage_df,
        "Fleet Status (fleet_status.csv)": fleet_status_df,
        "Oil Market (oil_market.csv)": oil_market_df,
    }
    table_choice = st.selectbox("Select a table to preview:", list(TABLE_OPTIONS.keys()))
    selected_df = TABLE_OPTIONS[table_choice]

    if selected_df is not None:
        n_rows, n_cols = selected_df.shape
        st.caption(f"{n_rows:,} rows × {n_cols} columns")

        relevant_cols = {
            k: v
            for k, v in FLEET_COLUMN_DESCRIPTIONS.items()
            if k in selected_df.columns
        }
        if relevant_cols:
            with st.expander("Column Descriptions", expanded=False):
                for col, desc in relevant_cols.items():
                    st.markdown(f"- **`{col}`** — {desc}")

        st.dataframe(
            selected_df.head(200),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("This table has not been generated yet. Run the pipeline scripts first.")

    st.divider()

    # --- Charts ---
    st.subheader("Waste Demand Analysis")

    if demand_df is not None and ports_df is not None:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**Average Daily Waste Volume by Port**")
            port_avg = (
                demand_df.groupby("port_code")["waste_volume_collected_m3"]
                .mean()
                .reset_index()
                .sort_values("waste_volume_collected_m3", ascending=True)
            )
            if "port_name" in ports_df.columns:
                port_avg = port_avg.merge(
                    ports_df[["port_code", "port_name"]], on="port_code", how="left"
                )
                label_col = "port_name"
            else:
                label_col = "port_code"

            fig_bar = px.bar(
                port_avg,
                x="waste_volume_collected_m3",
                y=label_col,
                orientation="h",
                labels={
                    "waste_volume_collected_m3": "Avg Daily Volume (m³)",
                    label_col: "Port",
                },
                color="waste_volume_collected_m3",
                color_continuous_scale="Blues",
            )
            fig_bar.update_layout(
                showlegend=False, coloraxis_showscale=False, margin=dict(l=0, r=0, t=20, b=0)
            )
            apply_plotly_theme(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            st.markdown("**Seasonal Demand Patterns (Monthly Average)**")
            demand_df_copy = demand_df.copy()
            demand_df_copy["month"] = demand_df_copy["date"].dt.month
            monthly = (
                demand_df_copy.groupby(["month", "port_code"])["waste_volume_collected_m3"]
                .mean()
                .reset_index()
            )
            top_ports = (
                demand_df_copy.groupby("port_code")["waste_volume_collected_m3"]
                .mean()
                .nlargest(5)
                .index.tolist()
            )
            monthly_top = monthly[monthly["port_code"].isin(top_ports)]
            fig_line = px.line(
                monthly_top,
                x="month",
                y="waste_volume_collected_m3",
                color="port_code",
                labels={
                    "month": "Month",
                    "waste_volume_collected_m3": "Avg Volume (m³)",
                    "port_code": "Port",
                },
                markers=True,
            )
            fig_line.update_xaxes(
                tickvals=list(range(1, 13)),
                ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            )
            fig_line.update_layout(margin=dict(l=0, r=0, t=20, b=0))
            apply_plotly_theme(fig_line)
            st.plotly_chart(fig_line, use_container_width=True)

        if fleet_status_df is not None and "current_cargo_pct" in fleet_status_df.columns:
            st.markdown("**Fleet Cargo Utilisation Over Time (rolling 30-day avg)**")
            daily_util = (
                fleet_status_df.assign(date=fleet_status_df["timestamp"].dt.date)
                .groupby("date")["current_cargo_pct"]
                .mean()
                .reset_index()
                .rename(columns={"current_cargo_pct": "avg_cargo_pct"})
            )
            daily_util["date"] = pd.to_datetime(daily_util["date"])
            daily_util["rolling_30d"] = daily_util["avg_cargo_pct"].rolling(30, min_periods=1).mean()
            fig_util = px.line(
                daily_util,
                x="date",
                y="rolling_30d",
                labels={"date": "Date", "rolling_30d": "Avg Cargo Utilisation (%)"},
            )
            fig_util.update_traces(line_color=COLORS["primary"])
            fig_util.update_layout(margin=dict(l=0, r=0, t=20, b=0))
            apply_plotly_theme(fig_util)
            st.plotly_chart(fig_util, use_container_width=True)


# ===========================================================================
# TAB 2 — ANALYTICS (NEW)
# ===========================================================================
with tab2:
    render_pipeline_steps(current_step=1, total_steps=5)
    st.divider()

    st.subheader("Fleet & Market Analytics")
    st.markdown("Descriptive analysis of port demand patterns, fleet performance, and market dynamics.")

    # Load data needed for analytics
    _demand_df = load_port_waste_demand()
    _voyages = load_voyage_history()
    _oil_df = load_oil_market()
    _ports_df_a = load_ports()

    if _demand_df is not None and _ports_df_a is not None:
        ports_list_a = sorted(_demand_df["port_code"].dropna().unique().tolist())
    elif _ports_df_a is not None:
        ports_list_a = sorted(_ports_df_a["port_code"].dropna().unique().tolist())
    else:
        ports_list_a = []

    # --- Port filter (global for this tab) ---
    port_filter = st.multiselect(
        "Filter by Port",
        ports_list_a,
        default=[],
        key="fleet_analytics_port",
    )

    st.divider()

    # ---- 1. Seasonal Demand Patterns ----------------------------------------
    st.markdown("#### Seasonal Demand Patterns")

    if _demand_df is not None:
        _dem = _demand_df.copy()
        if port_filter:
            _dem = _dem[_dem["port_code"].isin(port_filter)]

        _dem["month"] = _dem["date"].dt.month
        _dem["month_name"] = _dem["date"].dt.strftime("%b")
        monthly_demand = (
            _dem.groupby(["month", "month_name", "port_code"])["waste_volume_collected_m3"]
            .mean()
            .reset_index()
            .sort_values("month")
        )

        # Use top ports for readability if no filter applied
        if not port_filter:
            _top5 = (
                _dem.groupby("port_code")["waste_volume_collected_m3"]
                .mean()
                .nlargest(5)
                .index.tolist()
            )
            monthly_demand = monthly_demand[monthly_demand["port_code"].isin(_top5)]

        fig_seasonal = px.line(
            monthly_demand,
            x="month",
            y="waste_volume_collected_m3",
            color="port_code",
            labels={
                "month": "Month",
                "waste_volume_collected_m3": "Avg Daily Volume (m³)",
                "port_code": "Port",
            },
            markers=True,
            title="Monthly Average Waste Volume per Port",
        )
        fig_seasonal.update_xaxes(
            tickvals=list(range(1, 13)),
            ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        )
        # Cruise season shaded band Apr–Oct (months 4–10)
        fig_seasonal.add_vrect(
            x0=4, x1=10,
            fillcolor="rgba(27,154,170,0.08)",
            layer="below",
            line_width=0,
            annotation_text="Cruise Season (Apr–Oct)",
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color="#1B9AAA",
        )
        apply_plotly_theme(fig_seasonal)
        st.plotly_chart(fig_seasonal, use_container_width=True)
    else:
        st.info("port_waste_demand.csv not found. Run `python scripts/generate_data.py`.")

    # ---- 2. Calendar Heatmap ------------------------------------------------
    st.markdown("#### Daily Demand Heatmap")

    if _demand_df is not None and ports_list_a:
        selected_port_hm = st.selectbox(
            "Select port for heatmap",
            ports_list_a,
            key="heatmap_port",
        )
        port_data_hm = _demand_df[_demand_df["port_code"] == selected_port_hm].copy()
        if not port_data_hm.empty:
            fig_cal = calendar_heatmap(
                port_data_hm,
                "date",
                "waste_volume_collected_m3",
                f"Daily Waste Volume — {selected_port_hm}",
            )
            apply_plotly_theme(fig_cal)
            st.plotly_chart(fig_cal, use_container_width=True)
        else:
            st.info(f"No data available for port {selected_port_hm}.")
    else:
        st.info("Demand data not available.")

    # ---- 3. Fleet Utilization Dashboard -------------------------------------
    st.markdown("#### Fleet Performance by Vessel")

    if _voyages is not None:
        _voy = _voyages.copy()
        if port_filter and "departure_port" in _voy.columns:
            _voy = _voy[_voy["departure_port"].isin(port_filter)]

        if not _voy.empty and "vessel_name" in _voy.columns:
            vessel_perf = (
                _voy.groupby("vessel_name")
                .agg(
                    voyages=("voyage_id", "count") if "voyage_id" in _voy.columns else ("vessel_name", "count"),
                    total_distance_nm=("distance_nm", "sum"),
                    total_revenue_eur=("revenue_eur", "sum"),
                    total_fuel_cost_eur=("fuel_cost_eur", "sum"),
                    avg_margin_eur=("voyage_margin_eur", "mean"),
                )
                .reset_index()
                .sort_values("avg_margin_eur", ascending=False)
            )

            fig_vessel = px.bar(
                vessel_perf,
                x="vessel_name",
                y="avg_margin_eur",
                color="avg_margin_eur",
                color_continuous_scale=["#C62828", "#F9A825", "#2E7D32"],
                labels={
                    "vessel_name": "Vessel",
                    "avg_margin_eur": "Avg Voyage Margin (€)",
                },
                title="Average Voyage Margin by Vessel (sorted descending)",
            )
            fig_vessel.update_layout(coloraxis_showscale=False)
            apply_plotly_theme(fig_vessel)
            st.plotly_chart(fig_vessel, use_container_width=True)

            with st.expander("Vessel Performance Table", expanded=False):
                st.dataframe(
                    vessel_perf.style.format({
                        "total_revenue_eur": "€{:,.0f}",
                        "total_fuel_cost_eur": "€{:,.0f}",
                        "avg_margin_eur": "€{:,.0f}",
                        "total_distance_nm": "{:,.0f} nm",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No voyage data available for the selected ports.")
    else:
        st.info("voyage_history.csv not found. Run `python scripts/generate_data.py`.")

    # ---- 4. Voyage Profitability Distribution --------------------------------
    st.markdown("#### Voyage Profitability")

    if _voyages is not None and "voyage_margin_eur" in _voyages.columns:
        _voy_full = _voyages.copy()
        group_col_prof = "vessel_class" if "vessel_class" in _voy_full.columns else None

        fig_profit = profitability_distribution(
            _voy_full,
            "voyage_margin_eur",
            group_col=group_col_prof,
            title="Voyage Margin Distribution by Vessel Class",
        )
        apply_plotly_theme(fig_profit)
        st.plotly_chart(fig_profit, use_container_width=True)

        loss_pct = (_voy_full["voyage_margin_eur"] < 0).mean() * 100
        st.metric("Loss-Making Voyages", f"{loss_pct:.1f}%")
    else:
        st.info("voyage_history.csv not found or missing voyage_margin_eur column.")

    # ---- 5. Fuel Cost Sensitivity -------------------------------------------
    st.markdown("#### Fuel Price Impact on Margins")

    if _voyages is not None and _oil_df is not None:
        _voy_dates = _voyages.copy()
        if "departure_time" in _voy_dates.columns:
            _voy_dates["date"] = _voy_dates["departure_time"].dt.normalize()
        elif "date" in _voy_dates.columns:
            _voy_dates["date"] = pd.to_datetime(_voy_dates["date"])

        if "date" in _voy_dates.columns and "brent_crude_usd_bbl" in _oil_df.columns:
            _oil_daily = _oil_df[["date", "brent_crude_usd_bbl"]].copy()
            merged_fuel = _voy_dates.merge(_oil_daily, on="date", how="left")
            merged_fuel = merged_fuel.dropna(subset=["brent_crude_usd_bbl", "voyage_margin_eur"])

            if not merged_fuel.empty:
                color_col_fuel = "vessel_class" if "vessel_class" in merged_fuel.columns else None
                _sample = merged_fuel.sample(min(3000, len(merged_fuel)), random_state=42)
                fig_scatter = scatter_with_trend(
                    _sample,
                    "brent_crude_usd_bbl",
                    "voyage_margin_eur",
                    color_col=color_col_fuel,
                    title="Brent Crude vs Voyage Margin",
                )
                apply_plotly_theme(fig_scatter)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Not enough overlapping dates between voyage and oil market data.")
        else:
            st.info("Date alignment between voyage history and oil market data not possible.")
    else:
        st.info("voyage_history.csv or oil_market.csv not found.")

    # ---- 6. Route Efficiency ------------------------------------------------
    st.markdown("#### Most Profitable Routes")

    if _voyages is not None:
        _rv = _voyages.copy()
        if "departure_port" in _rv.columns and "arrival_port" in _rv.columns:
            _rv["route"] = _rv["departure_port"] + " → " + _rv["arrival_port"]
            _rv["revenue_per_nm"] = np.where(
                _rv["distance_nm"] > 0,
                _rv["revenue_eur"] / _rv["distance_nm"],
                np.nan,
            ) if "revenue_eur" in _rv.columns and "distance_nm" in _rv.columns else np.nan

            route_eff = (
                _rv.groupby("route")
                .agg(
                    voyages=("route", "count"),
                    avg_revenue_per_nm=("revenue_per_nm", "mean"),
                    avg_margin_eur=("voyage_margin_eur", "mean"),
                )
                .reset_index()
                .sort_values("avg_revenue_per_nm", ascending=False)
                .head(20)
            )

            fig_route = px.bar(
                route_eff,
                x="avg_revenue_per_nm",
                y="route",
                orientation="h",
                color="avg_margin_eur",
                color_continuous_scale=["#C62828", "#F9A825", "#2E7D32"],
                labels={
                    "avg_revenue_per_nm": "Avg Revenue per NM (€/nm)",
                    "route": "Route",
                    "avg_margin_eur": "Avg Margin (€)",
                },
                title="Top 20 Routes by Revenue per Nautical Mile",
            )
            fig_route.update_layout(yaxis={"categoryorder": "total ascending"})
            apply_plotly_theme(fig_route)
            st.plotly_chart(fig_route, use_container_width=True)
        else:
            st.info("departure_port / arrival_port columns not found in voyage_history.")
    else:
        st.info("voyage_history.csv not found.")

    # ---- 7. Market Impact (Dual Axis) ---------------------------------------
    st.markdown("#### Market Dynamics vs Fleet Margins")

    if _voyages is not None and _oil_df is not None:
        _voy_m = _voyages.copy()
        if "departure_time" in _voy_m.columns:
            _voy_m["month"] = _voy_m["departure_time"].dt.to_period("M").dt.to_timestamp()
        elif "date" in _voy_m.columns:
            _voy_m["month"] = pd.to_datetime(_voy_m["date"]).dt.to_period("M").dt.to_timestamp()

        if "month" in _voy_m.columns and "voyage_margin_eur" in _voy_m.columns:
            monthly_margin = (
                _voy_m.groupby("month")["voyage_margin_eur"]
                .mean()
                .reset_index()
                .rename(columns={"voyage_margin_eur": "avg_margin_eur"})
            )

            _oil_m = _oil_df.copy()
            _oil_m["month"] = _oil_m["date"].dt.to_period("M").dt.to_timestamp()
            monthly_brent = (
                _oil_m.groupby("month")["brent_crude_usd_bbl"]
                .mean()
                .reset_index()
                .rename(columns={"brent_crude_usd_bbl": "avg_brent_usd"})
            )

            monthly_combined = monthly_margin.merge(monthly_brent, on="month", how="inner")
            monthly_combined = monthly_combined.sort_values("month")

            if not monthly_combined.empty:
                fig_dual = dual_axis_chart(
                    monthly_combined,
                    "month",
                    "avg_margin_eur",
                    "avg_brent_usd",
                    "Monthly Avg Margin (€)",
                    "Brent Crude ($/bbl)",
                    "Fleet Margins vs Oil Price",
                )
                apply_plotly_theme(fig_dual)
                st.plotly_chart(fig_dual, use_container_width=True)
            else:
                st.info("No overlapping monthly data between voyages and oil market.")
        else:
            st.info("Could not compute monthly margins from voyage history.")
    else:
        st.info("voyage_history.csv or oil_market.csv not found.")

    # ---- 8. Port Performance ------------------------------------------------
    st.markdown("#### Port Revenue & Volume Comparison")

    if _voyages is not None and _demand_df is not None:
        # Waste volume per port from demand data
        port_vol = (
            _demand_df.groupby("port_code")["waste_volume_collected_m3"]
            .sum()
            .reset_index()
            .rename(columns={"waste_volume_collected_m3": "total_waste_m3"})
        )

        # Revenue per departure port from voyages
        if "departure_port" in _voyages.columns and "revenue_eur" in _voyages.columns:
            port_rev = (
                _voyages.groupby("departure_port")["revenue_eur"]
                .sum()
                .reset_index()
                .rename(columns={"departure_port": "port_code", "revenue_eur": "total_revenue_eur"})
            )
            port_perf = port_vol.merge(port_rev, on="port_code", how="left")
            port_perf["total_revenue_eur"] = port_perf["total_revenue_eur"].fillna(0)

            if port_filter:
                port_perf = port_perf[port_perf["port_code"].isin(port_filter)]

            fig_port = go.Figure()
            fig_port.add_trace(go.Bar(
                name="Total Waste Collected (m³)",
                x=port_perf["port_code"],
                y=port_perf["total_waste_m3"],
                marker_color="#1B9AAA",
                yaxis="y",
                hovertemplate="<b>%{x}</b><br>Waste: %{y:,.0f} m³<extra></extra>",
            ))
            fig_port.add_trace(go.Bar(
                name="Total Revenue (€)",
                x=port_perf["port_code"],
                y=port_perf["total_revenue_eur"],
                marker_color="#F9A825",
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>Revenue: €%{y:,.0f}<extra></extra>",
            ))
            fig_port.update_layout(
                title="Port Revenue & Waste Volume Comparison",
                barmode="group",
                yaxis=dict(title="Total Waste (m³)", gridcolor="#E0E4E8"),
                yaxis2=dict(
                    title="Total Revenue (€)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            apply_plotly_theme(fig_port)
            st.plotly_chart(fig_port, use_container_width=True)
        else:
            st.info("departure_port / revenue_eur columns not found in voyage_history.")
    else:
        st.info("voyage_history.csv or port_waste_demand.csv not found.")


# ===========================================================================
# TAB 3 — DEMAND FORECAST PIPELINE (Feature Engineering)
# ===========================================================================
with tab3:
    render_pipeline_steps(current_step=2, total_steps=5)
    st.divider()

    st.subheader("Demand Forecast Pipeline — Feature Engineering")

    fleet_features_df = load_fleet_features()
    route_features_df = load_route_features()

    if fleet_features_df is None:
        st.warning(
            "fleet_features.csv not found. Run `python scripts/engineer_features.py`."
        )
        st.stop()

    n_rows, n_cols = fleet_features_df.shape
    c1, c2, c3 = st.columns(3)
    c1.metric("Feature Matrix Rows", f"{n_rows:,}")
    c2.metric("Feature Columns", n_cols)
    c3.metric("Ports", fleet_features_df["port_code"].nunique())

    st.divider()

    # --- Data flow explanation ---
    _demand_len = len(load_port_waste_demand()) if load_port_waste_demand() is not None else 0
    _voyage_len = len(load_voyage_history()) if load_voyage_history() is not None else 0
    _oil_len = len(load_oil_market()) if load_oil_market() is not None else 0
    _weather_len = len(load_maritime_weather()) if load_maritime_weather() is not None else 0

    render_data_flow_explanation(
        source_tables=[
            ("port_waste_demand.csv", _demand_len, "Daily waste volume per port"),
            ("voyage_history.csv", _voyage_len, "Individual voyage records"),
            ("oil_market.csv", _oil_len, "Brent crude & fuel price series"),
            ("maritime_weather.csv", _weather_len, "Weather zone conditions"),
        ],
        features_added=[
            ("vessel_calls_7d_rolling", "7-day rolling mean of daily vessel arrivals"),
            ("waste_7d_rolling", "7-day rolling mean of collected waste volume"),
            ("waste_30d_rolling", "30-day rolling mean — captures longer trend"),
            ("yoy_growth_pct", "(current − same_period_last_year) / same_period_last_year"),
            ("cruise_season_flag", "1 if month ∈ [Apr–Oct] AND port is Mediterranean"),
            ("urgency_score", "current_storage_fill_pct × (1 / days_until_full)"),
        ],
        model_name="XGBoost Demand Forecaster\n(n_estimators=200, max_depth=6, lr=0.08)",
        result_description="Daily waste volume forecast per port with ±20% confidence band",
    )

    st.divider()

    # --- RAW vs ENGINEERED columns ---
    st.subheader("Column Taxonomy — Raw vs Engineered")

    RAW_COLS = [
        "demand_id", "date", "port_code", "port_name", "vessel_calls_total",
        "waste_volume_collected_m3", "current_storage_fill_pct", "days_until_full",
        "hec_market_share_pct", "avg_vessel_size_gt",
    ]
    ENGINEERED_COLS = [
        "vessel_calls_7d_rolling", "vessel_calls_30d_rolling",
        "waste_7d_rolling", "waste_30d_rolling",
        "yoy_growth_pct", "cruise_season_flag",
        "urgency_score", "storage_fill_rate_m3_day",
        "waste_per_vessel_call", "port_congestion_proxy",
        "oil_price_trend_7d", "brent_price_7d_avg",
    ]

    raw_present = [c for c in RAW_COLS if c in fleet_features_df.columns]
    eng_present = [c for c in ENGINEERED_COLS if c in fleet_features_df.columns]

    col_raw, col_eng = st.columns(2)
    with col_raw:
        st.markdown("**📋 Raw / Source Columns**")
        st.dataframe(
            fleet_features_df[raw_present].head(50),
            use_container_width=True,
            hide_index=True,
        )
    with col_eng:
        st.markdown("**⚙️ Engineered Feature Columns**")
        st.dataframe(
            fleet_features_df[eng_present].head(50),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # --- Feature engineering logic walkthrough ---
    st.subheader("Feature Engineering Logic")

    with st.expander("Rolling Window Features", expanded=True):
        st.markdown(
            """
| Feature | Formula | Business Meaning |
|---|---|---|
| `vessel_calls_7d_rolling` | 7-day rolling mean of `vessel_calls_total` per port | Short-term shipping traffic trend |
| `waste_7d_rolling` | 7-day rolling mean of `waste_volume_collected_m3` per port | Recent demand baseline |
| `waste_30d_rolling` | 30-day rolling mean of `waste_volume_collected_m3` per port | Monthly demand trend |
| `oil_price_trend_7d` | 7-day rolling mean of Brent crude price | Shipping activity proxy |
            """
        )

    with st.expander("Seasonal & Growth Features", expanded=True):
        st.markdown(
            """
| Feature | Formula | Business Meaning |
|---|---|---|
| `yoy_growth_pct` | (current − same_period_last_year) / same_period_last_year × 100 | Year-over-year demand growth |
| `cruise_season_flag` | 1 if month ∈ [4..10] AND port is in Mediterranean | Cruise traffic multiplier |
| `storage_fill_rate_m3_day` | (current fill − previous fill) / days elapsed | How fast port storage fills |
            """
        )

    with st.expander("Urgency & Risk Features", expanded=True):
        st.markdown(
            """
| Feature | Formula | Business Meaning |
|---|---|---|
| `urgency_score` | `current_storage_fill_pct` × (1 / `days_until_full`) | Composite urgency for collection scheduling |
| `port_congestion_proxy` | `vessel_calls_total` / `avg_daily_vessel_calls` | Port above or below normal traffic |
| `waste_per_vessel_call` | `waste_volume_collected_m3` / `vessel_calls_total` | Average waste generated per vessel |
| `market_share_risk` | Competitor presence flag × (1 − `hec_market_share_pct`/100) | Risk of losing waste contract |
            """
        )

    st.divider()

    # --- Correlation matrix ---
    st.subheader("Feature Correlation with Target (waste_volume_collected_m3)")

    target_col = "waste_volume_collected_m3"
    numeric_features = [
        c for c in fleet_features_df.columns
        if fleet_features_df[c].dtype in [np.float64, np.int64, float, int]
        and c != target_col
        and c not in ("port_latitude", "port_longitude")
    ]
    sample_df = fleet_features_df[numeric_features + [target_col]].dropna().sample(
        min(5000, len(fleet_features_df)), random_state=42
    )
    correlations = (
        sample_df.corr()[target_col]
        .drop(target_col)
        .abs()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )
    correlations.columns = ["Feature", "Abs Correlation"]

    fig_corr = px.bar(
        correlations,
        x="Abs Correlation",
        y="Feature",
        orientation="h",
        color="Abs Correlation",
        color_continuous_scale="Blues",
        labels={"Abs Correlation": "|Correlation| with Target"},
    )
    fig_corr.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis={"categoryorder": "total ascending"},
    )
    apply_plotly_theme(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # --- Feature distributions by port ---
    st.subheader("Feature Distributions by Port")

    dist_feature = st.selectbox(
        "Select feature to visualise:",
        [c for c in eng_present if c in fleet_features_df.columns],
        key="tab3_feature_dist",
    )
    selected_ports_t3 = st.multiselect(
        "Filter ports (blank = all):",
        options=sorted(fleet_features_df["port_code"].unique()),
        default=[],
        key="tab3_port_filter",
    )

    dist_df = fleet_features_df.copy()
    if selected_ports_t3:
        dist_df = dist_df[dist_df["port_code"].isin(selected_ports_t3)]

    if dist_feature and dist_feature in dist_df.columns:
        fig_box = px.box(
            dist_df.dropna(subset=[dist_feature]),
            x="port_code",
            y=dist_feature,
            color="port_code",
            labels={"port_code": "Port", dist_feature: dist_feature},
        )
        fig_box.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
        apply_plotly_theme(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)


# ===========================================================================
# TAB 4 — ML MODEL PERFORMANCE
# ===========================================================================
with tab4:
    render_pipeline_steps(current_step=3, total_steps=5)
    st.divider()

    st.subheader("ML Model Performance")

    demand_metrics = load_demand_metrics()
    predictions_df = load_demand_predictions()
    port_metrics_df = load_demand_port_metrics()
    profit_metrics = load_profitability_metrics()
    profit_importances = load_profitability_feature_importances()
    demand_model_artifact = load_demand_model()

    # --- Model A: Demand Forecaster ---
    st.markdown("### Model A: XGBoost Demand Forecaster")
    st.markdown(
        "Predicts daily waste volume (m³) per port. "
        "Trained on fleet_features.csv using time-based split (train < 2025-07-01)."
    )

    if demand_metrics is not None:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("MAPE", f"{demand_metrics['mape']:.1f}%", help="Mean Absolute Percentage Error — lower is better")
        mc2.metric("RMSE", f"{demand_metrics['rmse']:.1f} m³", help="Root Mean Squared Error")
        mc3.metric("R²", f"{demand_metrics['r2']:.3f}", help="Coefficient of determination — 1.0 = perfect")
        mc4.metric("Test Rows", f"{demand_metrics.get('test_rows', 'N/A'):,}")
    else:
        st.warning("Model metrics not found. Run `python scripts/train_models.py`.")

    st.divider()

    # --- Feature importance ---
    if demand_model_artifact is not None:
        st.markdown("#### Feature Importance — Demand Forecaster")
        try:
            model_obj = demand_model_artifact.get("model")
            feature_cols = demand_model_artifact.get("feature_cols", [])
            if model_obj is not None and hasattr(model_obj, "feature_importances_"):
                importances = model_obj.feature_importances_
                imp_df = (
                    pd.DataFrame({"Feature": feature_cols, "Importance": importances})
                    .sort_values("Importance", ascending=False)
                    .head(20)
                )
                fig_imp = px.bar(
                    imp_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Blues",
                )
                fig_imp.update_layout(
                    showlegend=False,
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=20, b=0),
                    yaxis={"categoryorder": "total ascending"},
                )
                apply_plotly_theme(fig_imp)
                st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.caption(f"Feature importance unavailable: {e}")

    # --- Actual vs Predicted scatter ---
    if predictions_df is not None and not predictions_df.empty:
        st.markdown("#### Actual vs Predicted Demand — Test Set (per port)")

        ports_in_preds = sorted(predictions_df["port_code"].unique())
        scatter_ports = st.multiselect(
            "Filter ports for scatter (blank = all):",
            options=ports_in_preds,
            default=[],
            key="tab4_scatter_ports",
        )
        scatter_df = predictions_df.copy()
        if scatter_ports:
            scatter_df = scatter_df[scatter_df["port_code"].isin(scatter_ports)]

        scatter_sample = scatter_df.sample(min(3000, len(scatter_df)), random_state=42)

        fig_scatter_t4 = px.scatter(
            scatter_sample,
            x="waste_volume_collected_m3",
            y="predicted_volume_m3",
            color="port_code",
            labels={
                "waste_volume_collected_m3": "Actual Volume (m³)",
                "predicted_volume_m3": "Predicted Volume (m³)",
                "port_code": "Port",
            },
            opacity=0.5,
        )
        max_val_t4 = max(
            scatter_sample["waste_volume_collected_m3"].max(),
            scatter_sample["predicted_volume_m3"].max(),
        )
        fig_scatter_t4.add_trace(
            go.Scatter(
                x=[0, max_val_t4],
                y=[0, max_val_t4],
                mode="lines",
                line=dict(color="red", dash="dash", width=1),
                name="Perfect Prediction",
                showlegend=True,
            )
        )
        fig_scatter_t4.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        apply_plotly_theme(fig_scatter_t4)
        st.plotly_chart(fig_scatter_t4, use_container_width=True)

    st.divider()

    # --- Time series: actual vs predicted ---
    st.markdown("#### Time Series — Actual vs Predicted for Selected Port")

    if predictions_df is not None and not predictions_df.empty:
        ts_port = st.selectbox(
            "Select port:",
            sorted(predictions_df["port_code"].unique()),
            key="tab4_ts_port",
        )
        port_ts = predictions_df[predictions_df["port_code"] == ts_port].sort_values("date")

        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=port_ts["date"],
                y=port_ts["waste_volume_collected_m3"],
                mode="lines",
                name="Actual",
                line=dict(color=COLORS["primary"]),
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=port_ts["date"],
                y=port_ts["predicted_volume_m3"],
                mode="lines",
                name="Predicted",
                line=dict(color=COLORS["secondary"], dash="dot"),
            )
        )
        fig_ts.update_layout(
            title=f"Port {ts_port} — Actual vs Predicted Waste Volume",
            xaxis_title="Date",
            yaxis_title="Volume (m³)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        apply_plotly_theme(fig_ts)
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Predictions file not found. Run `python scripts/train_models.py`.")

    st.divider()

    # --- Per-port metrics table ---
    if port_metrics_df is not None and not port_metrics_df.empty:
        st.markdown("#### Per-Port Accuracy Metrics")
        st.dataframe(
            port_metrics_df.sort_values("mape").style.format(
                {"mape": "{:.2f}%", "rmse": "{:.2f}", "r2": "{:.3f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # --- Model B: Profitability Model ---
    st.markdown("### Model B: Voyage Profitability Model")
    st.markdown(
        "Predicts voyage margin (€) to inform route economic viability decisions. "
        "Trained on voyage_history.csv feature matrix."
    )

    if profit_metrics is not None:
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("R²", f"{profit_metrics.get('r2', 'N/A'):.3f}")
        pc2.metric("RMSE", f"€{profit_metrics.get('rmse', 0):,.0f}")
        pc3.metric("MAE", f"€{profit_metrics.get('mae', 0):,.0f}")
    else:
        st.info("Profitability model metrics not found. Run `python scripts/train_models.py`.")

    if profit_importances is not None and not profit_importances.empty:
        st.markdown("#### Feature Importance — Profitability Model")
        if "feature" in profit_importances.columns and "importance" in profit_importances.columns:
            pimp = profit_importances.sort_values("importance", ascending=False).head(15)
        else:
            pimp = profit_importances.head(15)

        if len(pimp.columns) >= 2:
            fcol, icol = pimp.columns[0], pimp.columns[1]
            fig_pimp = px.bar(
                pimp,
                x=icol,
                y=fcol,
                orientation="h",
                color=icol,
                color_continuous_scale="Oranges",
                labels={icol: "Importance", fcol: "Feature"},
            )
            fig_pimp.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis={"categoryorder": "total ascending"},
            )
            apply_plotly_theme(fig_pimp)
            st.plotly_chart(fig_pimp, use_container_width=True)


# ===========================================================================
# TAB 5 — FLEET OPTIMIZATION DASHBOARD
# ===========================================================================
with tab5:
    render_pipeline_steps(current_step=4, total_steps=5)
    st.divider()

    st.subheader("Fleet Optimization Dashboard")
    st.markdown(
        "Select a date to see ML-predicted port demand, current fleet status, "
        "and the optimized vessel-to-port assignment plan."
    )

    # Load required data
    ports_df_t5 = load_ports()
    demand_df_t5 = load_port_waste_demand()
    fleet_status_t5 = load_fleet_status()
    oil_market_t5 = load_oil_market()
    weather_t5 = load_maritime_weather()
    voyage_df_t5 = load_voyage_history()

    if ports_df_t5 is None or demand_df_t5 is None or fleet_status_t5 is None:
        st.warning(
            "Required data files missing. Run `python scripts/generate_data.py` first."
        )
        st.stop()

    # --- Date selector ---
    date_min_t5 = demand_df_t5["date"].min().date()
    date_max_t5 = demand_df_t5["date"].max().date()
    selected_date_t5 = st.date_input(
        "Select optimization date:",
        value=date_max_t5,
        min_value=date_min_t5,
        max_value=date_max_t5,
        key="tab5_date",
    )
    selected_ts_t5 = pd.Timestamp(selected_date_t5)

    # --- Port demand snapshot ---
    demand_snap_t5 = demand_df_t5[demand_df_t5["date"] == selected_ts_t5].copy()
    if demand_snap_t5.empty:
        closest_t5 = demand_df_t5.loc[(demand_df_t5["date"] - selected_ts_t5).abs().idxmin()]
        selected_ts_t5 = closest_t5["date"]
        demand_snap_t5 = demand_df_t5[demand_df_t5["date"] == selected_ts_t5].copy()
        st.info(f"No data for selected date — showing closest available: {selected_ts_t5.date()}")

    if "urgency_score" not in demand_snap_t5.columns:
        demand_snap_t5["urgency_score"] = (
            demand_snap_t5["current_storage_fill_pct"].fillna(50) / 100.0
            * (1.0 / demand_snap_t5["days_until_full"].clip(lower=1).fillna(7))
        )

    # --- Fleet snapshot ---
    fleet_snap_t5 = fleet_status_t5[
        fleet_status_t5["timestamp"].dt.normalize() <= selected_ts_t5
    ].copy()
    if fleet_snap_t5.empty:
        fleet_snap_t5 = fleet_status_t5.copy()
    fleet_snap_t5 = fleet_snap_t5.sort_values("timestamp").groupby("vessel_id").last().reset_index()

    # --- Demand map ---
    st.subheader(f"Predicted Port Demand — {selected_ts_t5.date()}")
    demand_map_t5, dh_t5 = render_port_map(
        ports_df_t5,
        demand_df=demand_snap_t5[
            ["port_code", "waste_volume_collected_m3", "current_storage_fill_pct", "urgency_score"]
        ],
        height=400,
    )
    st_folium(demand_map_t5, height=dh_t5, width=None)
    st.caption(
        "Circle size = forecast waste volume. Color: green < 50% fill, orange 50–75%, red > 75% fill."
    )

    st.divider()

    # --- Naive baseline computation ---
    # Naive = average distance * assumed fuel rate gives a baseline cost estimate
    _avg_distance_nm = 0.0
    _avg_fuel_cost_per_voyage = 0.0
    if voyage_df_t5 is not None and "distance_nm" in voyage_df_t5.columns and "fuel_cost_eur" in voyage_df_t5.columns:
        _avg_distance_nm = voyage_df_t5["distance_nm"].mean()
        _avg_fuel_cost_per_voyage = voyage_df_t5["fuel_cost_eur"].mean()
        _avg_margin_historical = voyage_df_t5["voyage_margin_eur"].mean() if "voyage_margin_eur" in voyage_df_t5.columns else 0.0
    else:
        _avg_distance_nm = 0.0
        _avg_fuel_cost_per_voyage = 0.0
        _avg_margin_historical = 0.0

    # --- Run optimization ---
    st.subheader("Greedy Fleet Assignment")

    run_optimization_t5 = st.button("Run Route Optimizer", type="primary", key="tab5_run_opt")

    if run_optimization_t5 or "tab5_assignments" in st.session_state:
        with st.spinner("Running greedy vessel-to-port assignment..."):
            try:
                from models.fleet.route_optimizer import optimize_fleet_assignments, compute_fleet_kpis

                market_snap_t5 = (
                    oil_market_t5[oil_market_t5["date"] <= selected_ts_t5].tail(1).copy()
                    if oil_market_t5 is not None
                    else pd.DataFrame()
                )
                weather_snap_t5 = (
                    weather_t5[weather_t5["timestamp"].dt.normalize() <= selected_ts_t5].tail(30).copy()
                    if weather_t5 is not None
                    else pd.DataFrame()
                )

                assignments_t5 = optimize_fleet_assignments(
                    fleet_snap_t5,
                    demand_snap_t5,
                    weather_snap_t5,
                    market_snap_t5,
                )
                kpis_t5 = compute_fleet_kpis(assignments_t5, fleet_snap_t5)
                st.session_state["tab5_assignments"] = assignments_t5
                st.session_state["tab5_kpis"] = kpis_t5

            except Exception as e:
                st.error(f"Optimizer error: {e}")
                assignments_t5 = pd.DataFrame()
                kpis_t5 = {}
                st.session_state["tab5_assignments"] = assignments_t5
                st.session_state["tab5_kpis"] = kpis_t5

        assignments_t5 = st.session_state.get("tab5_assignments", pd.DataFrame())
        kpis_t5 = st.session_state.get("tab5_kpis", {})

        if not assignments_t5.empty:
            # --- Fleet KPIs ---
            st.subheader("Fleet KPIs")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Vessels Assigned", assignments_t5["vessel_id"].nunique())
            k2.metric(
                "Fleet Utilisation",
                f"{kpis_t5.get('fleet_utilization_pct', 0):.1f}%",
            )
            k3.metric(
                "Total Expected Revenue",
                f"€{kpis_t5.get('total_expected_revenue', 0):,.0f}",
            )
            k4.metric(
                "Total Fuel Cost",
                f"€{kpis_t5.get('total_fuel_cost', 0):,.0f}",
            )
            k5.metric(
                "SLA Compliance",
                f"{kpis_t5.get('sla_compliance_pct', 0):.1f}%",
            )

            st.divider()

            # --- Optimization vs Naive comparison ---
            st.subheader("Optimization vs Naive Dispatch")

            n_vessels_assigned = assignments_t5["vessel_id"].nunique()
            opt_total_fuel = kpis_t5.get("total_fuel_cost", 0.0)
            opt_total_revenue = kpis_t5.get("total_expected_revenue", 0.0)
            opt_total_margin = opt_total_revenue - opt_total_fuel

            # Naive baseline: random dispatch = n vessels × avg historical fuel cost
            naive_total_fuel = n_vessels_assigned * _avg_fuel_cost_per_voyage
            naive_total_margin = n_vessels_assigned * _avg_margin_historical

            margin_improvement = opt_total_margin - naive_total_margin
            fuel_savings = naive_total_fuel - opt_total_fuel

            cmp1, cmp2 = st.columns(2)
            with cmp1:
                st.markdown("**Naive (Random Dispatch)**")
                st.metric("Est. Fuel Cost", f"€{naive_total_fuel:,.0f}")
                st.metric("Est. Net Margin", f"€{naive_total_margin:,.0f}")
            with cmp2:
                st.markdown("**Optimized (ML-Guided)**")
                st.metric("Est. Fuel Cost", f"€{opt_total_fuel:,.0f}", delta=f"€{fuel_savings:+,.0f} vs naive")
                st.metric("Est. Net Margin", f"€{opt_total_margin:,.0f}", delta=f"€{margin_improvement:+,.0f} vs naive")

            if margin_improvement > 0:
                render_money_callout(
                    "Route Optimization Daily Savings",
                    margin_improvement,
                    f"ML-guided assignment generates €{margin_improvement:,.0f} more net margin "
                    f"than random dispatch across {n_vessels_assigned} assigned vessels.",
                )
            else:
                st.info(
                    "Optimization margin improvement not yet computable — "
                    "run full historical voyage baseline to compare."
                )

            st.divider()

            # --- Route map with lines ---
            st.subheader("Optimized Route Map")
            route_tuples_t5 = []
            port_lookup_t5 = ports_df_t5.set_index("port_code")
            for _, row in assignments_t5.iterrows():
                vessel_row_t5 = fleet_snap_t5[fleet_snap_t5["vessel_id"] == row["vessel_id"]]
                if not vessel_row_t5.empty:
                    dest_t5 = vessel_row_t5.iloc[0].get("destination_port", "")
                    if dest_t5 and dest_t5 in port_lookup_t5.index:
                        route_tuples_t5.append((dest_t5, row["assigned_port"], row["vessel_name"]))

            route_map_t5, rh_t5 = render_port_map(
                ports_df_t5,
                demand_df=demand_snap_t5[
                    ["port_code", "waste_volume_collected_m3", "current_storage_fill_pct", "urgency_score"]
                ],
                routes=route_tuples_t5 if route_tuples_t5 else None,
                height=440,
            )
            st_folium(route_map_t5, height=rh_t5, width=None)
            st.caption("Orange lines = assigned vessel routes. Circle size = forecast demand.")

            st.divider()

            # --- Schedule table ---
            st.subheader("Route Schedule")

            schedule_display_t5 = assignments_t5[
                [
                    c for c in [
                        "vessel_name", "assigned_port", "distance_nm",
                        "eta_hours", "expected_volume_m3",
                        "expected_revenue_eur", "fuel_cost_eur",
                        "expected_margin_eur", "urgency_score", "score",
                    ]
                    if c in assignments_t5.columns
                ]
            ].copy()

            if "port_name" in ports_df_t5.columns:
                pnames_t5 = ports_df_t5[["port_code", "port_name"]].set_index("port_code")["port_name"]
                schedule_display_t5.insert(
                    2,
                    "port_name",
                    schedule_display_t5["assigned_port"].map(pnames_t5).fillna(
                        schedule_display_t5["assigned_port"]
                    ),
                )

            if "eta_hours" in schedule_display_t5.columns:
                schedule_display_t5["eta_display"] = schedule_display_t5["eta_hours"].apply(
                    lambda h: f"{int(h // 24)}d {int(h % 24)}h" if pd.notna(h) else "—"
                )

            # Highlight negative margins
            schedule_sorted_t5 = schedule_display_t5.sort_values(
                "urgency_score", ascending=False
            )

            def _highlight_loss(row):
                if "expected_margin_eur" in row.index and pd.notna(row.get("expected_margin_eur")):
                    if row["expected_margin_eur"] < 0:
                        return ["background-color: #FFEBEE"] * len(row)
                return [""] * len(row)

            fmt_dict = {}
            if "expected_revenue_eur" in schedule_sorted_t5.columns:
                fmt_dict["expected_revenue_eur"] = "€{:,.0f}"
            if "fuel_cost_eur" in schedule_sorted_t5.columns:
                fmt_dict["fuel_cost_eur"] = "€{:,.0f}"
            if "expected_margin_eur" in schedule_sorted_t5.columns:
                fmt_dict["expected_margin_eur"] = "€{:,.0f}"
            if "distance_nm" in schedule_sorted_t5.columns:
                fmt_dict["distance_nm"] = "{:,.0f}"
            if "urgency_score" in schedule_sorted_t5.columns:
                fmt_dict["urgency_score"] = "{:.3f}"

            st.dataframe(
                schedule_sorted_t5.style.apply(_highlight_loss, axis=1).format(fmt_dict),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Red rows indicate voyages with negative expected margin.")

        else:
            st.info("No assignments generated. Check that fleet status and demand data are available.")

    st.divider()

    # --- 7-day demand forecast ---
    st.subheader("7-Day Demand Forecast by Port")

    pred_df_t5 = load_demand_predictions()
    if pred_df_t5 is not None and not pred_df_t5.empty:
        forecast_start_t5 = selected_ts_t5
        forecast_end_t5 = selected_ts_t5 + pd.Timedelta(days=6)
        forecast_window_t5 = pred_df_t5[
            (pred_df_t5["date"] >= forecast_start_t5) & (pred_df_t5["date"] <= forecast_end_t5)
        ].copy()

        if forecast_window_t5.empty:
            forecast_window_t5 = pred_df_t5.sort_values("date").groupby("port_code").tail(7).copy()
            st.caption("Showing last 7 available prediction days (selected date outside test set).")

        if not forecast_window_t5.empty:
            all_ports_fc_t5 = sorted(forecast_window_t5["port_code"].unique())
            sel_ports_fc_t5 = st.multiselect(
                "Select ports to display (blank = all):",
                options=all_ports_fc_t5,
                default=[],
                key="tab5_forecast_ports",
            )
            fc_df_t5 = forecast_window_t5.copy()
            if sel_ports_fc_t5:
                fc_df_t5 = fc_df_t5[fc_df_t5["port_code"].isin(sel_ports_fc_t5)]

            fig_fc_t5 = px.line(
                fc_df_t5.sort_values("date"),
                x="date",
                y="predicted_volume_m3",
                color="port_code",
                labels={
                    "date": "Date",
                    "predicted_volume_m3": "Forecast Volume (m³)",
                    "port_code": "Port",
                },
                markers=True,
                title="7-Day Waste Demand Forecast",
            )
            fig_fc_t5.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=50, b=0),
            )
            apply_plotly_theme(fig_fc_t5)
            st.plotly_chart(fig_fc_t5, use_container_width=True)
    else:
        st.info(
            "Demand forecast predictions not found. Run `python scripts/train_models.py`."
        )

    st.divider()

    # --- Fleet status table ---
    st.subheader("Current Fleet Status")
    if fleet_snap_t5 is not None and not fleet_snap_t5.empty:
        fleet_display_cols_t5 = [
            c for c in [
                "vessel_id", "vessel_name", "vessel_class",
                "status", "destination_port",
                "current_cargo_pct", "fuel_remaining_mt",
                "days_until_next_maintenance",
            ]
            if c in fleet_snap_t5.columns
        ]
        st.dataframe(
            fleet_snap_t5[fleet_display_cols_t5],
            use_container_width=True,
            hide_index=True,
        )
