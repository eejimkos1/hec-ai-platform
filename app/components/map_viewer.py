"""Map visualization component for HEC fleet routing.

Uses folium + streamlit-folium to render interactive maps of port locations,
demand overlays, and optimized route lines.
"""

from __future__ import annotations

import folium
import pandas as pd
from streamlit_folium import st_folium


def render_port_map(
    ports_df: pd.DataFrame,
    demand_df: pd.DataFrame | None = None,
    routes: list[tuple[str, str, str]] | None = None,
    height: int = 500,
):
    """Render an interactive map of HEC ports with optional demand overlay and routes.

    Args:
        ports_df: DataFrame with port_code, port_name, latitude, longitude
        demand_df: Optional - latest demand per port (adds circle markers sized by volume)
        routes:    Optional - list of (from_port_code, to_port_code, vessel_name) tuples
        height:    Map display height in pixels

    Returns:
        (folium.Map, int) — pass both to st_folium(map, height=height, width=None)
    """
    # Center on Mediterranean / Northern Europe overlap
    m = folium.Map(location=[38.0, 15.0], zoom_start=4, tiles="CartoDB positron")

    # -----------------------------------------------------------------------
    # Port markers
    # -----------------------------------------------------------------------
    for _, port in ports_df.iterrows():
        popup_html = (
            f"<b>{port['port_name']}</b><br>"
            f"Code: {port['port_code']}<br>"
            f"Country: {port.get('country', 'N/A')}"
        )
        lat = float(port["latitude"])
        lon = float(port["longitude"])

        if demand_df is not None and port["port_code"] in demand_df["port_code"].values:
            port_demand = demand_df[demand_df["port_code"] == port["port_code"]].iloc[0]
            volume = float(port_demand.get("waste_volume_collected_m3", 50))
            popup_html += f"<br>Daily Demand: {volume:.0f} m³"
            fill_pct = float(port_demand.get("current_storage_fill_pct", 0))
            urgency = float(port_demand.get("urgency_score", 0))
            if fill_pct:
                popup_html += f"<br>Storage Fill: {fill_pct:.0f}%"
            if urgency:
                popup_html += f"<br>Urgency: {urgency:.3f}"
            # Size by volume, clamp to [5, 25]
            radius = max(5, min(25, volume / 10))
            # Color by urgency: green → orange → red
            color = "#2ca02c" if fill_pct < 50 else ("#ff7f0e" if fill_pct < 75 else "#d62728")
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=folium.Popup(popup_html, max_width=200),
            ).add_to(m)
        else:
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=200),
                icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
            ).add_to(m)

    # -----------------------------------------------------------------------
    # Route lines
    # -----------------------------------------------------------------------
    if routes:
        port_lookup = ports_df.set_index("port_code")
        for from_code, to_code, vessel in routes:
            if from_code not in port_lookup.index or to_code not in port_lookup.index:
                continue
            from_port = port_lookup.loc[from_code]
            to_port = port_lookup.loc[to_code]
            folium.PolyLine(
                locations=[
                    [float(from_port["latitude"]), float(from_port["longitude"])],
                    [float(to_port["latitude"]), float(to_port["longitude"])],
                ],
                color="#ff7f0e",
                weight=2.5,
                opacity=0.8,
                popup=f"{vessel}: {from_code} → {to_code}",
            ).add_to(m)

    return m, height


def render_demand_heatmap(
    ports_df: pd.DataFrame,
    demand_summary_df: pd.DataFrame,
    height: int = 450,
):
    """Render a map where circle size represents total demand volume at each port.

    Args:
        ports_df:           DataFrame with port_code, port_name, latitude, longitude
        demand_summary_df:  DataFrame with port_code, total_demand (m³)
        height:             Map display height in pixels

    Returns:
        (folium.Map, int)
    """
    m = folium.Map(location=[38.0, 15.0], zoom_start=4, tiles="CartoDB positron")

    if demand_summary_df.empty or "total_demand" not in demand_summary_df.columns:
        return m, height

    max_demand = demand_summary_df["total_demand"].max()

    for _, port in ports_df.iterrows():
        code = port["port_code"]
        if code not in demand_summary_df["port_code"].values:
            continue
        total = float(
            demand_summary_df[demand_summary_df["port_code"] == code]["total_demand"].iloc[0]
        )
        # Scale radius 5–30 based on fraction of max demand
        radius = 5 + 25 * (total / max(max_demand, 1))
        folium.CircleMarker(
            location=[float(port["latitude"]), float(port["longitude"])],
            radius=radius,
            color="#d62728",
            fill=True,
            fill_color="#d62728",
            fill_opacity=0.5,
            popup=f"{port['port_name']}: {total:,.0f} m³ total",
        ).add_to(m)

    return m, height
