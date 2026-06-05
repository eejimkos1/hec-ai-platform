"""Reusable Plotly chart helpers for HEC AI Platform pages.

All charts use the HEC corporate color palette (navy/teal) and return go.Figure objects
ready to drop into any Streamlit page via st.plotly_chart().
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# HEC Corporate Color Palette & Layout
# Defined locally to avoid circular imports with app/config.py
# ---------------------------------------------------------------------------

HEC_COLORWAY = [
    "#0D1B2A",  # navy
    "#1B9AAA",  # teal
    "#2E7D32",  # forest green
    "#5C6BC0",  # indigo
    "#F9A825",  # amber
    "#C62828",  # crimson
    "#78909C",  # blue-grey
    "#4DB6AC",  # light teal
]

HEC_LAYOUT = dict(
    font=dict(
        family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        size=12,
    ),
    colorway=HEC_COLORWAY,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor="#0D1B2A", font_color="white", font_size=12),
    margin=dict(t=40, b=40, l=50, r=20),
)

# Shared axis style helpers
_GRID_STYLE = dict(showgrid=True, gridcolor="#E0E4E8", gridwidth=1)


def _apply_hec(fig: go.Figure, height: int = None) -> go.Figure:
    """Apply HEC layout + grid lines to a figure and return it."""
    layout_kwargs = dict(**HEC_LAYOUT)
    if height is not None:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(**_GRID_STYLE)
    fig.update_yaxes(**_GRID_STYLE)
    return fig


# ---------------------------------------------------------------------------
# Existing functions — updated with HEC palette & rich hover
# ---------------------------------------------------------------------------


def yield_distribution_chart(
    df: pd.DataFrame,
    color_col: str = "facility_code",
    colors: dict = None,
) -> go.Figure:
    """Histogram of oil recovery yield by facility (or other grouping column)."""
    fig = px.histogram(
        df,
        x="oil_recovery_yield_pct",
        color=color_col,
        nbins=50,
        opacity=0.75,
        barmode="overlay",
        color_discrete_map=colors,
        color_discrete_sequence=HEC_COLORWAY,
        labels={"oil_recovery_yield_pct": "Oil Recovery Yield (%)"},
    )
    fig.update_traces(
        hovertemplate="<b>%{x:.1f}%</b><br>Count: %{y}<extra></extra>"
    )
    _apply_hec(fig, height=350)
    fig.update_layout(legend_title_text=color_col)
    return fig


def feature_importance_chart(
    model,
    feature_names: list,
    top_n: int = 15,
) -> go.Figure:
    """Horizontal bar chart of top-N feature importances from a trained model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    fig = go.Figure(
        go.Bar(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            orientation="h",
            marker_color=HEC_COLORWAY[1],  # teal
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    _apply_hec(fig, height=max(350, top_n * 22 + 60))
    fig.update_layout(
        margin=dict(l=220, t=40, b=40, r=20),
        xaxis_title="Importance Score",
        yaxis_title="",
    )
    return fig


def actual_vs_predicted_chart(
    y_actual: pd.Series,
    y_predicted: np.ndarray,
) -> go.Figure:
    """Scatter plot of actual vs predicted yield values with a perfect-fit line."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_actual,
            y=y_predicted,
            mode="markers",
            marker=dict(size=5, opacity=0.45, color=HEC_COLORWAY[1]),
            name="Predictions",
            hovertemplate=(
                "<b>Actual:</b> %{x:.2f}%<br>"
                "<b>Predicted:</b> %{y:.2f}%<extra></extra>"
            ),
        )
    )
    min_val = float(min(y_actual.min(), y_predicted.min()))
    max_val = float(max(y_actual.max(), y_predicted.max()))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(dash="dash", color=HEC_COLORWAY[5], width=1.5),
            name="Perfect fit",
            hoverinfo="skip",
        )
    )
    _apply_hec(fig, height=420)
    fig.update_layout(
        xaxis_title="Actual Yield (%)",
        yaxis_title="Predicted Yield (%)",
    )
    return fig


def residual_distribution_chart(residuals: np.ndarray) -> go.Figure:
    """Histogram of model residuals (actual minus predicted)."""
    fig = px.histogram(
        x=residuals,
        nbins=60,
        opacity=0.78,
        color_discrete_sequence=[HEC_COLORWAY[4]],  # amber
        labels={"x": "Residual (pp)"},
    )
    fig.update_traces(
        hovertemplate="<b>Residual:</b> %{x:.2f} pp<br>Count: %{y}<extra></extra>"
    )
    fig.add_vline(x=0, line_dash="dash", line_color=HEC_COLORWAY[5], line_width=1.5)
    _apply_hec(fig, height=300)
    fig.update_layout(showlegend=False)
    return fig


def time_series_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str = None,
    title: str = "",
) -> go.Figure:
    """Line chart for time-series data."""
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        color_discrete_sequence=HEC_COLORWAY,
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Value: %{y:,.1f}<extra></extra>"
    )
    _apply_hec(fig, height=350)
    return fig


def optimization_comparison_chart(
    before_params: list,
    after_params: list,
    param_names: list,
) -> go.Figure:
    """Grouped bar chart comparing current vs optimised parameter values."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Current (baseline)",
            x=param_names,
            y=before_params,
            marker_color=HEC_COLORWAY[4],  # amber
            hovertemplate="<b>%{x}</b><br>Current: %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Optimised",
            x=param_names,
            y=after_params,
            marker_color=HEC_COLORWAY[1],  # teal
            hovertemplate="<b>%{x}</b><br>Optimised: %{y:,.2f}<extra></extra>",
        )
    )
    _apply_hec(fig, height=380)
    fig.update_layout(
        barmode="group",
        yaxis_title="Parameter Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def correlation_heatmap(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 20,
) -> go.Figure:
    """Heatmap of Pearson correlations between top-N features and the target."""
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        return go.Figure()

    corr_with_target = numeric_df.corr()[target_col].drop(target_col).abs()
    top_features = corr_with_target.nlargest(top_n).index.tolist()
    sub = numeric_df[top_features + [target_col]]
    corr_matrix = sub.corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            showscale=True,
            hovertemplate=(
                "<b>%{y}</b> × <b>%{x}</b><br>"
                "Correlation: %{z:.3f}<extra></extra>"
            ),
        )
    )
    _apply_hec(fig, height=max(400, (top_n + 1) * 26 + 80))
    fig.update_layout(
        margin=dict(t=40, b=80, l=160, r=20),
        xaxis_tickangle=-45,
    )
    return fig


def waste_volume_by_facility_chart(
    df: pd.DataFrame,
    colors: dict = None,
) -> go.Figure:
    """Bar chart: total waste volume received per facility."""
    grp = (
        df.groupby("facility_code")["actual_volume_m3"]
        .sum()
        .reset_index()
        .rename(columns={"actual_volume_m3": "Total Volume (m³)"})
    )
    fig = px.bar(
        grp,
        x="facility_code",
        y="Total Volume (m³)",
        color="facility_code",
        color_discrete_map=colors,
        color_discrete_sequence=HEC_COLORWAY,
        labels={"facility_code": "Facility"},
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Volume: %{y:,.0f} m³<extra></extra>"
    )
    _apply_hec(fig, height=320)
    fig.update_layout(showlegend=False)
    return fig


def waste_category_pie_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart of waste volume by subcategory."""
    grp = (
        df.groupby("waste_subcategory")["actual_volume_m3"]
        .sum()
        .reset_index()
    )
    fig = px.pie(
        grp,
        names="waste_subcategory",
        values="actual_volume_m3",
        hole=0.35,
        color_discrete_sequence=HEC_COLORWAY,
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Volume: %{value:,.0f} m³<br>Share: %{percent}<extra></extra>"
    )
    _apply_hec(fig, height=340)
    fig.update_layout(legend=dict(font_size=10))
    return fig


def confusion_matrix_chart(
    cm: list,
    labels: list = None,
) -> go.Figure:
    """Annotated heatmap of a confusion matrix."""
    if labels is None:
        labels = ["Fail", "Pass"]
    z = np.array(cm)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"Predicted: {lbl}" for lbl in labels],
            y=[f"Actual: {lbl}" for lbl in labels],
            colorscale=[
                [0, "#FFFFFF"],
                [1, HEC_COLORWAY[1]],
            ],
            text=z,
            texttemplate="%{text}",
            showscale=False,
            hovertemplate=(
                "%{y}<br>%{x}<br>Count: %{z}<extra></extra>"
            ),
        )
    )
    _apply_hec(fig, height=280)
    fig.update_layout(margin=dict(t=40, b=40, l=100, r=20))
    return fig


# ---------------------------------------------------------------------------
# NEW functions for Analytics tabs
# ---------------------------------------------------------------------------


def violin_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    title: str = "",
    colors: list = None,
) -> go.Figure:
    """Violin plot comparing value distributions across groups.

    Parameters
    ----------
    df : DataFrame
    value_col : numeric column to plot on the y-axis
    group_col : categorical column that defines the groups
    title : chart title
    colors : optional list of hex colours; defaults to HEC_COLORWAY
    """
    palette = colors if colors else HEC_COLORWAY
    groups = df[group_col].dropna().unique().tolist()

    fig = go.Figure()
    for idx, group in enumerate(groups):
        subset = df[df[group_col] == group][value_col].dropna()
        fig.add_trace(
            go.Violin(
                y=subset,
                name=str(group),
                box_visible=True,
                meanline_visible=True,
                fillcolor=palette[idx % len(palette)],
                line_color=palette[idx % len(palette)],
                opacity=0.75,
                hovertemplate=(
                    f"<b>{group}</b><br>"
                    "Value: %{y:,.2f}<extra></extra>"
                ),
            )
        )

    _apply_hec(fig, height=420)
    fig.update_layout(
        title=title,
        yaxis_title=value_col,
        violinmode="group",
        showlegend=True,
    )
    return fig


def time_series_with_trend(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str = None,
    title: str = "",
    show_trend: bool = True,
) -> go.Figure:
    """Line chart with optional linear trend line, range slider, and range selector buttons.

    Buttons: 1m, 6m, 1y, All.
    """
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        color_discrete_sequence=HEC_COLORWAY,
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Value: %{y:,.1f}<extra></extra>"
    )

    # Add linear trend line when no colour grouping
    if show_trend and color_col is None:
        valid = df[[x_col, y_col]].dropna()
        if len(valid) >= 2:
            x_numeric = np.arange(len(valid))
            coeffs = np.polyfit(x_numeric, valid[y_col].values, 1)
            trend_y = np.polyval(coeffs, x_numeric)
            fig.add_trace(
                go.Scatter(
                    x=valid[x_col],
                    y=trend_y,
                    mode="lines",
                    name="Trend",
                    line=dict(dash="dot", color=HEC_COLORWAY[5], width=1.5),
                    hoverinfo="skip",
                )
            )

    _apply_hec(fig, height=380)
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
    )
    return fig


def radar_chart(
    categories: list,
    values_dict: dict,
    title: str = "",
) -> go.Figure:
    """Multi-series radar / spider chart.

    Parameters
    ----------
    categories : list of category labels (spokes)
    values_dict : {"Series A": [v1, v2, ...], "Series B": [v1, v2, ...]}
    title : chart title
    """
    fig = go.Figure()
    for idx, (series_name, vals) in enumerate(values_dict.items()):
        # Close the polygon by repeating the first value
        closed_cats = list(categories) + [categories[0]]
        closed_vals = list(vals) + [vals[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=closed_vals,
                theta=closed_cats,
                fill="toself",
                name=series_name,
                line_color=HEC_COLORWAY[idx % len(HEC_COLORWAY)],
                fillcolor=HEC_COLORWAY[idx % len(HEC_COLORWAY)],
                opacity=0.35,
                hovertemplate=(
                    f"<b>{series_name}</b><br>"
                    "%{theta}: %{r:,.2f}<extra></extra>"
                ),
            )
        )

    _apply_hec(fig, height=440)
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                gridcolor="#E0E4E8",
            ),
            angularaxis=dict(gridcolor="#E0E4E8"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def calendar_heatmap(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str = "",
) -> go.Figure:
    """Calendar-style heatmap.

    Y-axis = day of week (0 = Monday … 6 = Sunday).
    X-axis = ISO week number.
    Cell colour intensity encodes the aggregated value.
    """
    tmp = df[[date_col, value_col]].dropna().copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp["week"] = tmp[date_col].dt.isocalendar().week.astype(int)
    tmp["dow"] = tmp[date_col].dt.dayofweek  # 0=Mon … 6=Sun

    agg = tmp.groupby(["week", "dow"])[value_col].mean().reset_index()

    weeks = sorted(agg["week"].unique())
    days = list(range(7))
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    z = np.full((7, len(weeks)), np.nan)
    week_idx = {w: i for i, w in enumerate(weeks)}
    for _, row in agg.iterrows():
        z[int(row["dow"]), week_idx[int(row["week"])]] = row[value_col]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(w) for w in weeks],
            y=day_labels,
            colorscale=[
                [0.0, "#FFFFFF"],
                [0.5, HEC_COLORWAY[1]],
                [1.0, HEC_COLORWAY[0]],
            ],
            showscale=True,
            hovertemplate=(
                "Week %{x} / %{y}<br>"
                "Avg value: %{z:,.2f}<extra></extra>"
            ),
        )
    )

    _apply_hec(fig, height=280)
    fig.update_layout(
        title=title,
        xaxis_title="ISO Week",
        yaxis_title="Day of Week",
    )
    return fig


def profitability_distribution(
    df: pd.DataFrame,
    margin_col: str,
    group_col: str = None,
    title: str = "",
) -> go.Figure:
    """Combined histogram + box plot (marginal) showing profit margin distribution.

    Uses px.histogram with marginal="box" for a compact combined view.
    Colour-coded by group if group_col is provided.
    """
    kwargs = dict(
        x=margin_col,
        marginal="box",
        nbins=40,
        opacity=0.78,
        color_discrete_sequence=HEC_COLORWAY,
        labels={margin_col: "Net Margin (€)"},
        title=title,
    )
    if group_col:
        kwargs["color"] = group_col

    fig = px.histogram(df, **kwargs)
    fig.update_traces(
        selector=dict(type="histogram"),
        hovertemplate="<b>Margin:</b> €%{x:,.0f}<br>Count: %{y}<extra></extra>",
    )
    _apply_hec(fig, height=420)
    return fig


def dual_axis_chart(
    df: pd.DataFrame,
    x_col: str,
    y1_col: str,
    y2_col: str,
    y1_name: str = "",
    y2_name: str = "",
    title: str = "",
) -> go.Figure:
    """Two y-axes overlaid on a single chart.

    Left axis → y1_col (line, teal).
    Right axis → y2_col (line, amber).
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y1_col],
            name=y1_name or y1_col,
            mode="lines",
            line=dict(color=HEC_COLORWAY[1], width=2),
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{y1_name or y1_col}: %{{y:,.1f}}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y2_col],
            name=y2_name or y2_col,
            mode="lines",
            line=dict(color=HEC_COLORWAY[4], width=2),
            yaxis="y2",
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{y2_name or y2_col}: %{{y:,.1f}}<extra></extra>"
            ),
        )
    )

    _apply_hec(fig, height=400)
    fig.update_layout(
        title=title,
        yaxis=dict(
            title=y1_name or y1_col,
            gridcolor="#E0E4E8",
        ),
        yaxis2=dict(
            title=y2_name or y2_col,
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def cost_breakdown_donut(
    labels: list,
    values: list,
    title: str = "",
    currency: str = "EUR",
) -> go.Figure:
    """Donut chart with currency values in hover tooltip.

    Parameters
    ----------
    labels  : list of cost-category names
    values  : list of numeric amounts (same length as labels)
    title   : chart title
    currency: currency symbol/code shown in hover (default "EUR")
    """
    symbol = "€" if currency.upper() == "EUR" else currency

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=HEC_COLORWAY),
            hovertemplate=(
                "<b>%{label}</b><br>"
                f"Amount: {symbol}%{{value:,.0f}}<br>"
                "Share: %{percent}<extra></extra>"
            ),
            textinfo="label+percent",
        )
    )

    _apply_hec(fig, height=380)
    fig.update_layout(
        title=title,
        legend=dict(orientation="v", x=1.02, y=0.5),
    )
    return fig


def waterfall_chart(
    items: list,
    title: str = "",
) -> go.Figure:
    """Waterfall chart showing revenue → costs → margin flow.

    Parameters
    ----------
    items : list of (label, value, type_str) tuples
            type_str must be one of "increase", "decrease", or "total"
    title : chart title

    Example
    -------
    items = [
        ("Revenue",          450_000, "total"),
        ("Processing Cost", -120_000, "decrease"),
        ("Logistics",        -30_000, "decrease"),
        ("Net Margin",       300_000, "total"),
    ]
    """
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    measures = [item[2] for item in items]

    # Map measure strings to Plotly waterfall measure types
    measure_map = {
        "increase": "relative",
        "decrease": "relative",
        "total": "total",
    }
    plotly_measures = [measure_map.get(m, "relative") for m in measures]

    # Colour each bar based on measure type
    increasing_color = HEC_COLORWAY[2]   # forest green
    decreasing_color = HEC_COLORWAY[5]   # crimson
    total_color = HEC_COLORWAY[0]        # navy

    fig = go.Figure(
        go.Waterfall(
            name="",
            orientation="v",
            measure=plotly_measures,
            x=labels,
            y=values,
            connector=dict(line=dict(color="#78909C", dash="dot")),
            increasing=dict(marker_color=increasing_color),
            decreasing=dict(marker_color=decreasing_color),
            totals=dict(marker_color=total_color),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Amount: €%{y:,.0f}<extra></extra>"
            ),
            texttemplate="€%{y:,.0f}",
            textposition="outside",
        )
    )

    _apply_hec(fig, height=420)
    fig.update_layout(
        title=title,
        yaxis_title="Amount (EUR)",
        showlegend=False,
    )
    return fig


def scatter_with_trend(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str = None,
    title: str = "",
    show_r2: bool = True,
) -> go.Figure:
    """Scatter plot with OLS trend line and optional R² annotation.

    Parameters
    ----------
    df        : source DataFrame
    x_col     : numeric column for the x-axis
    y_col     : numeric column for the y-axis
    color_col : optional categorical column for point colouring
    title     : chart title
    show_r2   : if True, annotate chart with R² value
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_sequence=HEC_COLORWAY,
        title=title,
        labels={x_col: x_col, y_col: y_col},
        opacity=0.65,
    )
    fig.update_traces(
        marker=dict(size=6),
        hovertemplate=(
            "<b>%{x:,.2f}</b><br>"
            "Value: %{y:,.2f}<extra></extra>"
        ),
    )

    # OLS trend line on full dataset (ignoring colour groups)
    valid = df[[x_col, y_col]].dropna()
    if len(valid) >= 2:
        x_vals = valid[x_col].values
        y_vals = valid[y_col].values
        coeffs = np.polyfit(x_vals, y_vals, 1)
        trend_y = np.polyval(coeffs, x_vals)

        # R² calculation
        ss_res = np.sum((y_vals - trend_y) ** 2)
        ss_tot = np.sum((y_vals - y_vals.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=trend_y,
                mode="lines",
                name="OLS Trend",
                line=dict(color=HEC_COLORWAY[5], dash="dash", width=1.5),
                hoverinfo="skip",
            )
        )

        if show_r2 and not np.isnan(r2):
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.04,
                text=f"R² = {r2:.3f}",
                showarrow=False,
                font=dict(size=12, color=HEC_COLORWAY[0]),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor=HEC_COLORWAY[1],
                borderwidth=1,
            )

    _apply_hec(fig, height=420)
    return fig
