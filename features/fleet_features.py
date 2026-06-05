"""
fleet_features.py
HEC AI Platform – Task 6: Fleet & Demand Feature Engineering

Functions:
  compute_demand_features    – Category E: per-port per-day demand features
  compute_route_features     – Category F: per (vessel, port, date) route features
  compute_economic_features  – Category G: economic features added to demand frame
  build_fleet_feature_matrix – Main pipeline: demand + economic → fleet_features.csv
  build_route_feature_matrix – Route pipeline → route_features.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap – allow running as a standalone script
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.generators.common import haversine_distance_nm, PORTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mediterranean ports for cruise_season_flag
MED_PORTS = {
    "PIR", "MLT", "GIB", "GEN", "MRS", "BCN",
    "ALG", "LIM", "ALE", "IST", "PSD", "TAN",
}

# Port → weather zone mapping
PORT_TO_ZONE = {
    # MED-E
    "PIR": "MED-E",
    "LIM": "MED-E",
    "ALE": "MED-E",
    "IST": "MED-E",
    "PSD": "MED-E",
    # MED-C
    "MLT": "MED-C",
    "GEN": "MED-C",
    # MED-W
    "MRS": "MED-W",
    "BCN": "MED-W",
    # ATL-GIB
    "GIB": "ATL-GIB",
    "ALG": "ATL-GIB",
    "TAN": "ATL-GIB",
    # NORTH-SEA
    "HAM": "NORTH-SEA",
    # CHANNEL
    "RTM": "CHANNEL",
}

# Avg €/m³ acceptance fee and recovery value fraction
ACCEPTANCE_FEE_EUR_M3 = 30.0
RECOVERY_FRACTION = 0.3
RECOVERY_PRICE_EUR_M3 = 400.0
AVG_SPEED_KNOTS = 12.0           # for fuel cost estimate in compute_economic_features
PORT_CHARGES_EUR = 1500.0        # fixed port charge per voyage
FUEL_PRICE_FALLBACK_USD_MT = 500.0  # fallback if market_df not available


# ===========================================================================
# Category E – Demand Prediction Features
# ===========================================================================

def compute_demand_features(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-port per-day demand prediction features (Category E).

    Parameters
    ----------
    demand_df : pd.DataFrame
        port_waste_demand raw data.
    weather_df : pd.DataFrame
        maritime_weather raw data (6-hourly, zone_id column).
    market_df : pd.DataFrame
        oil_market raw data (business days, brent_crude_usd_bbl column).

    Returns
    -------
    pd.DataFrame
        Input columns preserved, plus engineered features.
    """
    df = demand_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ------------------------------------------------------------------ #
    # 1. Sort for rolling operations                                       #
    # ------------------------------------------------------------------ #
    df = df.sort_values(["port_code", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 2. Rolling features on vessel calls & waste volume (per port)       #
    # ------------------------------------------------------------------ #
    def _rolling(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=1).mean()

    grp = df.groupby("port_code", group_keys=False)

    df["vessel_calls_7d_rolling"] = grp["vessel_calls_total"].transform(
        lambda s: _rolling(s, 7)
    )
    df["vessel_calls_30d_rolling"] = grp["vessel_calls_total"].transform(
        lambda s: _rolling(s, 30)
    )

    df["waste_7d_rolling"] = grp["waste_volume_collected_m3"].transform(
        lambda s: _rolling(s, 7)
    )
    df["waste_30d_rolling"] = grp["waste_volume_collected_m3"].transform(
        lambda s: _rolling(s, 30)
    )

    # ------------------------------------------------------------------ #
    # 3. Waste per vessel call                                             #
    # ------------------------------------------------------------------ #
    df["waste_per_vessel_call"] = (
        df["waste_volume_collected_m3"]
        / df["vessel_calls_total"].clip(lower=1)
    )

    # ------------------------------------------------------------------ #
    # 4. Year-over-year growth                                             #
    # ------------------------------------------------------------------ #
    # Create a lookup: port + same day last year
    yoy_key = df[["port_code", "date", "waste_volume_collected_m3"]].copy()
    yoy_key["date_next_year"] = yoy_key["date"] + pd.DateOffset(years=1)
    yoy_key = yoy_key.rename(
        columns={"waste_volume_collected_m3": "waste_prev_year"}
    )[["port_code", "date_next_year", "waste_prev_year"]]
    yoy_key = yoy_key.rename(columns={"date_next_year": "date"})

    df = df.merge(yoy_key, on=["port_code", "date"], how="left")
    df["yoy_growth_pct"] = (
        (df["waste_volume_collected_m3"] - df["waste_prev_year"])
        / df["waste_prev_year"].clip(lower=1)
        * 100
    )
    df = df.drop(columns=["waste_prev_year"])

    # ------------------------------------------------------------------ #
    # 5. Cruise season flag                                                #
    # ------------------------------------------------------------------ #
    df["cruise_season_flag"] = (
        df["date"].dt.month.isin([4, 5, 6, 7, 8, 9, 10])
        & df["port_code"].isin(MED_PORTS)
    ).astype(int)

    # ------------------------------------------------------------------ #
    # 6. Port congestion proxy                                             #
    #    = vessel_calls_total / avg_daily_vessel_calls (from reference)   #
    # ------------------------------------------------------------------ #
    # avg_daily_vessel_calls is a static reference value loaded by the
    # caller (build_fleet_feature_matrix) which merges it into demand_df.
    # If the column already exists we use it; otherwise fall back to
    # vessel_calls_7d_rolling as the reference average.
    if "avg_daily_vessel_calls" in df.columns:
        df["port_congestion_proxy"] = (
            df["vessel_calls_total"]
            / df["avg_daily_vessel_calls"].clip(lower=1)
        )
    else:
        df["port_congestion_proxy"] = (
            df["vessel_calls_total"]
            / df["vessel_calls_7d_rolling"].clip(lower=1)
        )

    # ------------------------------------------------------------------ #
    # 7. Days since last collection                                        #
    #    Storage fill drops when HEC collects → we detect the fall by     #
    #    looking at current_storage_fill_pct differences.                 #
    # ------------------------------------------------------------------ #
    def _days_since_drop(s: pd.Series) -> pd.Series:
        """Count of rows since fill_pct last *decreased* (i.e. a collection)."""
        result = np.zeros(len(s), dtype=float)
        counter = 0
        prev = np.nan
        for i, v in enumerate(s):
            if not np.isnan(prev) and v < prev:
                counter = 0
            else:
                counter += 1
            result[i] = counter
            prev = v
        return pd.Series(result, index=s.index)

    df["days_since_last_collection"] = grp[
        "current_storage_fill_pct"
    ].transform(_days_since_drop)

    # ------------------------------------------------------------------ #
    # 8. Storage fill rate (m³/day)                                        #
    # ------------------------------------------------------------------ #
    if "port_reception_facility_capacity_m3" in df.columns:
        capacity_col = "port_reception_facility_capacity_m3"
    else:
        capacity_col = None

    if capacity_col:
        df["storage_fill_rate_m3_day"] = (
            grp["current_storage_fill_pct"]
            .transform(lambda s: s.diff().fillna(0))
            * df[capacity_col]
            / 100
        )
    else:
        df["storage_fill_rate_m3_day"] = 0.0

    # ------------------------------------------------------------------ #
    # 9. Weather window probability                                        #
    #    Fraction of zone weather records in ±3 days with feasibility=Go  #
    # ------------------------------------------------------------------ #
    # Map each port to its zone
    df["weather_zone"] = df["port_code"].map(PORT_TO_ZONE)

    # Aggregate weather to daily (fraction Go per zone per day)
    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["timestamp"]).dt.normalize()
    weather_df["is_go"] = (weather_df["operation_feasibility"] == "Go").astype(float)
    weather_daily = (
        weather_df.groupby(["zone_id", "date"])["is_go"]
        .mean()
        .reset_index(name="go_fraction")
    )

    # For each port-date, average the go_fraction over ±3 days
    # Build a zone×date lookup, then rolling-mean over 7 rows (±3d window = 7 days)
    weather_daily = weather_daily.sort_values(["zone_id", "date"])

    weather_window_7d = (
        weather_daily
        .groupby("zone_id")["go_fraction"]
        .transform(lambda s: s.rolling(7, min_periods=1, center=True).mean())
    )
    weather_daily["weather_window_probability"] = weather_window_7d.values

    # Merge into df
    df = df.merge(
        weather_daily[["zone_id", "date", "weather_window_probability"]].rename(
            columns={"zone_id": "weather_zone"}
        ),
        on=["weather_zone", "date"],
        how="left",
    )

    # Forward-fill missing weather data
    df = df.sort_values(["port_code", "date"])
    df["weather_window_probability"] = (
        df.groupby("port_code")["weather_window_probability"]
        .transform(lambda s: s.ffill().bfill().fillna(0.5))
    )

    # ------------------------------------------------------------------ #
    # 10. Market features: brent_price_7d_avg, shipping_demand_index       #
    # ------------------------------------------------------------------ #
    market_df = market_df.copy()
    market_df["date"] = pd.to_datetime(market_df["date"])
    market_daily = (
        market_df[["date", "brent_crude_usd_bbl", "shipping_demand_index"]]
        .sort_values("date")
        .drop_duplicates("date")
    )

    # Forward-fill to calendar days (market data is business-days only)
    full_dates = pd.date_range(market_daily["date"].min(), market_daily["date"].max(), freq="D")
    market_daily = (
        market_daily.set_index("date")
        .reindex(full_dates)
        .ffill()
        .reset_index()
        .rename(columns={"index": "date"})
    )

    market_daily["brent_price_7d_avg"] = (
        market_daily["brent_crude_usd_bbl"].rolling(7, min_periods=1).mean()
    )

    df = df.merge(
        market_daily[["date", "brent_price_7d_avg", "shipping_demand_index"]],
        on="date",
        how="left",
    )

    # Forward-fill any remaining gaps in market columns
    df["brent_price_7d_avg"] = (
        df.groupby("port_code")["brent_price_7d_avg"]
        .transform(lambda s: s.ffill().bfill())
    )
    df["shipping_demand_index"] = (
        df.groupby("port_code")["shipping_demand_index"]
        .transform(lambda s: s.ffill().bfill())
    )

    # ------------------------------------------------------------------ #
    # 11. Final sort                                                        #
    # ------------------------------------------------------------------ #
    df = df.sort_values(["port_code", "date"]).reset_index(drop=True)

    return df


# ===========================================================================
# Category F – Route Optimization Features
# ===========================================================================

def compute_route_features(
    fleet_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    distance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-(vessel, port, date) route optimization features (Category F).

    Samples ~50,000 rows: 100 random dates × all vessel×port combinations
    (sampled down to ≤50,000 total).

    Parameters
    ----------
    fleet_df : pd.DataFrame
        fleet_status raw data with vessel snapshots.
    demand_df : pd.DataFrame
        port_waste_demand raw data.
    distance_df : pd.DataFrame
        port_distance_matrix reference data.

    Returns
    -------
    pd.DataFrame
        One row per (vessel_id, port_code, date) with route features.
    """
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------ #
    # Prep fleet data                                                      #
    # ------------------------------------------------------------------ #
    fleet_df = fleet_df.copy()
    fleet_df["timestamp"] = pd.to_datetime(fleet_df["timestamp"])
    fleet_df["date"] = fleet_df["timestamp"].dt.normalize()

    # Get latest snapshot per vessel per date
    fleet_daily = (
        fleet_df.sort_values("timestamp")
        .groupby(["vessel_id", "date"])
        .last()
        .reset_index()
    )

    # ------------------------------------------------------------------ #
    # Prep demand data                                                     #
    # ------------------------------------------------------------------ #
    demand_df = demand_df.copy()
    demand_df["date"] = pd.to_datetime(demand_df["date"])

    # days_since_last_collection in demand: recompute cheaply from fill_pct drop
    demand_df = demand_df.sort_values(["port_code", "date"])
    demand_df["fill_diff"] = demand_df.groupby("port_code")["current_storage_fill_pct"].diff()
    demand_df["days_since_coll"] = (
        demand_df.groupby("port_code")["fill_diff"]
        .transform(lambda s: _cumcount_since_drop(s))
    )

    # Port coordinates from PORTS constant
    port_coords = {k: (v["lat"], v["lon"]) for k, v in PORTS.items()}

    # ------------------------------------------------------------------ #
    # Build distance lookup                                                #
    # ------------------------------------------------------------------ #
    dist_lookup: dict = {}
    if distance_df is not None and not distance_df.empty:
        for _, row in distance_df.iterrows():
            dist_lookup[(row["origin_port"], row["destination_port"])] = float(
                row["distance_nm"]
            )

    def _get_dist(origin_lat, origin_lon, dest_code):
        """Distance in nm from a lat/lon to a port."""
        if dest_code in port_coords:
            dlat, dlon = port_coords[dest_code]
            return haversine_distance_nm(origin_lat, origin_lon, dlat, dlon)
        return 200.0  # fallback

    # ------------------------------------------------------------------ #
    # Sample 100 random dates that exist in both fleet and demand          #
    # ------------------------------------------------------------------ #
    fleet_dates = fleet_daily["date"].dt.normalize().unique()
    demand_dates = demand_df["date"].dt.normalize().unique()
    common_dates = np.intersect1d(fleet_dates, demand_dates)

    if len(common_dates) == 0:
        return pd.DataFrame()

    n_sample_dates = min(100, len(common_dates))
    idx = rng.choice(len(common_dates), size=n_sample_dates, replace=False)
    sampled_dates = pd.DatetimeIndex(common_dates[idx])

    all_port_codes = list(demand_df["port_code"].unique())

    records = []

    for date_val in sampled_dates:
        # Fleet snapshot for this date
        fleet_snap = fleet_daily[fleet_daily["date"] == date_val]
        if fleet_snap.empty:
            continue

        # Demand state for this date
        demand_snap = demand_df[demand_df["date"] == date_val].set_index("port_code")

        for _, vessel in fleet_snap.iterrows():
            vid = vessel["vessel_id"]
            vlat = vessel["latitude"]
            vlon = vessel["longitude"]
            tank_cap = vessel["tank_capacity_m3"]
            cur_cargo = vessel.get("current_cargo_m3", 0.0)
            fuel_remain = vessel.get("fuel_remaining_mt", 0.0)
            fuel_rate = vessel.get("fuel_consumption_mt_day", 6.0)

            vessel_avail_cap = max(0.0, tank_cap - cur_cargo)
            fuel_autonomy = fuel_remain / max(fuel_rate, 0.1)

            for port_code in all_port_codes:
                if port_code not in demand_snap.index:
                    continue
                d_row = demand_snap.loc[port_code]

                # Distance
                proximity_nm = _get_dist(vlat, vlon, port_code)

                # Port demand state
                waste_vol = float(d_row.get("waste_volume_collected_m3", 0.0))
                fill_pct = float(d_row.get("current_storage_fill_pct", 50.0))
                days_until_full = float(d_row.get("days_until_full", 7.0))
                days_since_coll = float(d_row.get("days_since_coll", 0.0))

                # Urgency
                urgency_score = (fill_pct / 100.0) * (1.0 / max(days_until_full, 1.0))

                # Collection efficiency
                coll_efficiency = waste_vol / max(proximity_nm, 1.0)

                # Fuel cost per m³
                fuel_cost_per_m3 = (
                    (proximity_nm / 24.0 * fuel_rate * FUEL_PRICE_FALLBACK_USD_MT)
                    / max(waste_vol, 1.0)
                )

                # Multi-stop opportunity: other ports within 100nm with fill>50%
                if port_code in port_coords:
                    p_lat, p_lon = port_coords[port_code]
                else:
                    p_lat, p_lon = vlat, vlon

                multi_stop = 0
                for other_code in all_port_codes:
                    if other_code == port_code:
                        continue
                    if other_code not in port_coords:
                        continue
                    o_lat, o_lon = port_coords[other_code]
                    d_other = haversine_distance_nm(p_lat, p_lon, o_lat, o_lon)
                    if d_other <= 100.0 and other_code in demand_snap.index:
                        other_fill = float(
                            demand_snap.loc[other_code].get("current_storage_fill_pct", 0.0)
                        )
                        if other_fill > 50.0:
                            multi_stop += 1

                # Contract obligation
                contract_obligation = max(0.0, 14.0 - days_since_coll)

                records.append({
                    "date": date_val,
                    "vessel_id": vid,
                    "port_code": port_code,
                    "vessel_proximity_to_demand_nm": round(proximity_nm, 1),
                    "vessel_available_capacity_m3": round(vessel_avail_cap, 1),
                    "urgency_score": round(urgency_score, 4),
                    "collection_efficiency_m3_per_nm": round(coll_efficiency, 4),
                    "fuel_cost_per_m3_collected": round(fuel_cost_per_m3, 4),
                    "multi_stop_opportunity": multi_stop,
                    "contract_obligation_days": round(contract_obligation, 1),
                    "vessel_fuel_autonomy_days": round(fuel_autonomy, 1),
                    # weather_risk_enroute is merged below
                })

    if not records:
        return pd.DataFrame()

    route_df = pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    # Weather risk en-route: max significant_wave_height per zone per date #
    # (We attach the zone of the destination port and look up max hs)      #
    # ------------------------------------------------------------------ #
    # We don't have weather_df here – caller (build_route_feature_matrix)
    # can merge. Provide NaN placeholder so the schema is stable.
    route_df["weather_risk_enroute"] = np.nan

    # ------------------------------------------------------------------ #
    # Sample down to ≤50,000 rows                                          #
    # ------------------------------------------------------------------ #
    if len(route_df) > 50_000:
        route_df = route_df.sample(n=50_000, random_state=42).reset_index(drop=True)

    return route_df


def _cumcount_since_drop(diff_series: pd.Series) -> pd.Series:
    """Return running count of rows since last negative diff (i.e. a collection)."""
    result = np.zeros(len(diff_series), dtype=float)
    counter = 0
    for i, v in enumerate(diff_series):
        if v < 0:
            counter = 0
        else:
            counter += 1
        result[i] = counter
    return pd.Series(result, index=diff_series.index)


# ===========================================================================
# Category G – Economic Features
# ===========================================================================

def compute_economic_features(
    demand_df: pd.DataFrame,
    market_df: pd.DataFrame,
    distance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute economic features (Category G) and add them to demand_df.

    Parameters
    ----------
    demand_df : pd.DataFrame
        Output of compute_demand_features (or raw demand if called standalone).
    market_df : pd.DataFrame
        oil_market raw data.
    distance_df : pd.DataFrame
        port_distance_matrix reference data.

    Returns
    -------
    pd.DataFrame
        demand_df with economic feature columns appended.
    """
    df = demand_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ------------------------------------------------------------------ #
    # Average distance per port (from distance matrix)                    #
    # ------------------------------------------------------------------ #
    if distance_df is not None and not distance_df.empty:
        avg_dist = (
            distance_df.groupby("origin_port")["distance_nm"]
            .mean()
            .rename("avg_distance_nm")
        )
        df = df.merge(
            avg_dist.reset_index().rename(columns={"origin_port": "port_code"}),
            on="port_code",
            how="left",
        )
        df["avg_distance_nm"] = df["avg_distance_nm"].fillna(
            distance_df["distance_nm"].mean()
        )
    else:
        df["avg_distance_nm"] = 300.0  # fallback

    # ------------------------------------------------------------------ #
    # Prep market data                                                     #
    # ------------------------------------------------------------------ #
    market_df = market_df.copy()
    market_df["date"] = pd.to_datetime(market_df["date"])

    full_dates = pd.date_range(market_df["date"].min(), market_df["date"].max(), freq="D")
    market_daily = (
        market_df[["date", "brent_crude_usd_bbl", "rotterdam_hfo_380_usd_mt"]]
        .sort_values("date")
        .drop_duplicates("date")
        .set_index("date")
        .reindex(full_dates)
        .ffill()
        .reset_index()
        .rename(columns={"index": "date"})
    )

    market_daily["brent_7d_ago"] = market_daily["brent_crude_usd_bbl"].shift(7)
    market_daily["oil_price_trend_7d"] = (
        (market_daily["brent_crude_usd_bbl"] - market_daily["brent_7d_ago"])
        / market_daily["brent_7d_ago"].clip(lower=1)
        * 100
    )

    df = df.merge(
        market_daily[["date", "rotterdam_hfo_380_usd_mt", "oil_price_trend_7d"]],
        on="date",
        how="left",
    )

    # Forward-fill missing market data
    df = df.sort_values(["port_code", "date"])
    for col in ["rotterdam_hfo_380_usd_mt", "oil_price_trend_7d"]:
        df[col] = (
            df.groupby("port_code")[col]
            .transform(lambda s: s.ffill().bfill())
        )

    # Fallback if still missing
    df["rotterdam_hfo_380_usd_mt"] = df["rotterdam_hfo_380_usd_mt"].fillna(
        FUEL_PRICE_FALLBACK_USD_MT
    )
    df["oil_price_trend_7d"] = df["oil_price_trend_7d"].fillna(0.0)

    # ------------------------------------------------------------------ #
    # Revenue calculation                                                  #
    # expected_revenue = waste_vol × 30 + waste_vol × 0.3 × 400           #
    # ------------------------------------------------------------------ #
    waste_vol = df["waste_volume_collected_m3"]
    df["expected_revenue_per_voyage"] = (
        waste_vol * ACCEPTANCE_FEE_EUR_M3
        + waste_vol * RECOVERY_FRACTION * RECOVERY_PRICE_EUR_M3
    )

    # ------------------------------------------------------------------ #
    # Cost calculation                                                     #
    # voyage_cost = (avg_dist_nm / 24 × 12 × fuel_price/1000) + 1500     #
    # ------------------------------------------------------------------ #
    fuel_usd_per_mt = df["rotterdam_hfo_380_usd_mt"]
    df["voyage_cost_estimate"] = (
        (df["avg_distance_nm"] / 24.0 * AVG_SPEED_KNOTS * fuel_usd_per_mt / 1000.0)
        + PORT_CHARGES_EUR
    )

    df["expected_margin"] = df["expected_revenue_per_voyage"] - df["voyage_cost_estimate"]

    # ------------------------------------------------------------------ #
    # Oil price trend already computed above                              #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Market share risk                                                    #
    # 1 if competitor_presence > 2 AND days_since_last_collection > 5    #
    # ------------------------------------------------------------------ #
    if "competitor_presence" in df.columns:
        # competitor_presence is boolean in raw data; competitor_count is int
        # We need competitor count > 2
        if df["competitor_presence"].dtype == bool or df["competitor_presence"].dtype == object:
            # Fall back to avg_daily_vessel_calls proxy if competitor_count absent
            if "competitor_count" in df.columns:
                comp_condition = df["competitor_count"] > 2
            else:
                # Boolean flag: treat True as 1 competitor, which cannot exceed 2
                comp_condition = pd.Series(False, index=df.index)
        else:
            comp_condition = df["competitor_presence"] > 2
    elif "competitor_count" in df.columns:
        comp_condition = df["competitor_count"] > 2
    else:
        comp_condition = pd.Series(False, index=df.index)

    days_since_col = "days_since_last_collection"
    if days_since_col in df.columns:
        dsc_condition = df[days_since_col] > 5
    elif "days_since_coll" in df.columns:
        dsc_condition = df["days_since_coll"] > 5
    else:
        # Fallback: never trigger
        dsc_condition = pd.Series(False, index=df.index)

    df["market_share_risk"] = (comp_condition & dsc_condition).astype(int)

    # Drop the temporary avg_distance column if it wasn't already present
    # (keep it if useful for downstream — left in place)

    return df


# ===========================================================================
# Main Pipelines
# ===========================================================================

def build_fleet_feature_matrix(
    raw_data_dir: str = "data/raw",
    reference_dir: str = "data/reference",
    output_path: str = "data/features/fleet_features.csv",
) -> pd.DataFrame:
    """
    Main pipeline: load raw data, compute demand + economic features,
    save to output_path, return the DataFrame.

    This is the primary output for Model 2A (demand / scheduling).

    Parameters
    ----------
    raw_data_dir : str
        Directory containing raw CSV files generated by fleet_data_generator.
    reference_dir : str
        Directory containing reference CSVs (ports.csv, port_distance_matrix.csv).
    output_path : str
        Destination CSV path.

    Returns
    -------
    pd.DataFrame
        Combined demand + economic feature matrix.
    """
    # ------------------------------------------------------------------ #
    # Resolve paths relative to repo root                                 #
    # ------------------------------------------------------------------ #
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir  = Path(raw_data_dir) if Path(raw_data_dir).is_absolute() else repo_root / raw_data_dir
    ref_dir  = Path(reference_dir) if Path(reference_dir).is_absolute() else repo_root / reference_dir
    out_path = Path(output_path) if Path(output_path).is_absolute() else repo_root / output_path

    # ------------------------------------------------------------------ #
    # Auto-generate raw data if missing                                   #
    # ------------------------------------------------------------------ #
    required_raw = [
        "port_waste_demand.csv",
        "maritime_weather.csv",
        "oil_market.csv",
    ]
    missing = [f for f in required_raw if not (raw_dir / f).exists()]
    if missing:
        print(f"Raw data files not found ({missing}). Generating now ...")
        from data.generators.fleet_data_generator import generate_all
        data = generate_all()
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name, df_gen in data.items():
            df_gen.to_csv(raw_dir / f"{name}.csv", index=False)
        print("Raw data generated and saved.")

    # ------------------------------------------------------------------ #
    # Load raw CSVs                                                        #
    # ------------------------------------------------------------------ #
    print("Loading raw CSVs ...")
    demand_df  = pd.read_csv(raw_dir / "port_waste_demand.csv")
    weather_df = pd.read_csv(raw_dir / "maritime_weather.csv")
    market_df  = pd.read_csv(raw_dir / "oil_market.csv")

    demand_df["date"]  = pd.to_datetime(demand_df["date"])
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
    market_df["date"]  = pd.to_datetime(market_df["date"])

    # ------------------------------------------------------------------ #
    # Load reference data                                                  #
    # ------------------------------------------------------------------ #
    print("Loading reference data ...")
    ports_df    = pd.read_csv(ref_dir / "ports.csv")
    distance_df = pd.read_csv(ref_dir / "port_distance_matrix.csv")

    # Merge avg_daily_vessel_calls + competitor_count from ports reference
    ports_ref = ports_df[["port_code", "avg_daily_vessel_calls", "competitor_count"]].copy()
    demand_df = demand_df.merge(ports_ref, on="port_code", how="left")

    # ------------------------------------------------------------------ #
    # Step 4: Compute demand features (Category E)                        #
    # ------------------------------------------------------------------ #
    print("Computing demand features (Category E) ...")
    demand_features_df = compute_demand_features(
        demand_df=demand_df,
        weather_df=weather_df,
        market_df=market_df,
    )

    # ------------------------------------------------------------------ #
    # Step 5: Compute economic features (Category G)                      #
    # ------------------------------------------------------------------ #
    print("Computing economic features (Category G) ...")
    full_df = compute_economic_features(
        demand_df=demand_features_df,
        market_df=market_df,
        distance_df=distance_df,
    )

    # ------------------------------------------------------------------ #
    # Step 6: Save                                                         #
    # ------------------------------------------------------------------ #
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"Saved fleet feature matrix to {out_path}  ({full_df.shape[0]:,} rows x {full_df.shape[1]} cols)")

    return full_df


def build_route_feature_matrix(
    raw_data_dir: str = "data/raw",
    reference_dir: str = "data/reference",
    output_path: str = "data/features/route_features.csv",
) -> pd.DataFrame:
    """
    Route feature pipeline: load raw data, compute route features (Category F),
    save to output_path.  Used by Model 2B / 2C.

    Parameters
    ----------
    raw_data_dir : str
        Directory containing raw CSV files.
    reference_dir : str
        Directory containing reference CSVs.
    output_path : str
        Destination CSV path.

    Returns
    -------
    pd.DataFrame
        Route feature matrix (~50,000 rows).
    """
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir  = Path(raw_data_dir) if Path(raw_data_dir).is_absolute() else repo_root / raw_data_dir
    ref_dir  = Path(reference_dir) if Path(reference_dir).is_absolute() else repo_root / reference_dir
    out_path = Path(output_path) if Path(output_path).is_absolute() else repo_root / output_path

    # Auto-generate if missing
    required_raw = ["fleet_status.csv", "port_waste_demand.csv"]
    missing = [f for f in required_raw if not (raw_dir / f).exists()]
    if missing:
        print(f"Raw data files not found ({missing}). Generating now ...")
        from data.generators.fleet_data_generator import generate_all
        data = generate_all()
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name, df_gen in data.items():
            df_gen.to_csv(raw_dir / f"{name}.csv", index=False)
        print("Raw data generated and saved.")

    print("Loading raw CSVs for route features ...")
    fleet_df  = pd.read_csv(raw_dir / "fleet_status.csv")
    demand_df = pd.read_csv(raw_dir / "port_waste_demand.csv")
    weather_df = pd.read_csv(raw_dir / "maritime_weather.csv")

    fleet_df["timestamp"] = pd.to_datetime(fleet_df["timestamp"])
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

    distance_df = pd.read_csv(ref_dir / "port_distance_matrix.csv")

    print("Computing route features (Category F) ...")
    route_df = compute_route_features(
        fleet_df=fleet_df,
        demand_df=demand_df,
        distance_df=distance_df,
    )

    if route_df.empty:
        print("Warning: route feature matrix is empty.")
        return route_df

    # ------------------------------------------------------------------ #
    # Merge weather risk en-route                                          #
    # ------------------------------------------------------------------ #
    weather_df["date"] = pd.to_datetime(weather_df["timestamp"]).dt.normalize()
    weather_max_hs = (
        weather_df.groupby(["zone_id", "date"])["significant_wave_height_m"]
        .max()
        .reset_index(name="max_hs")
    )

    route_df["weather_zone"] = route_df["port_code"].map(PORT_TO_ZONE)
    route_df = route_df.merge(
        weather_max_hs.rename(columns={"zone_id": "weather_zone", "max_hs": "weather_risk_enroute_new"}),
        on=["weather_zone", "date"],
        how="left",
    )
    # Use the merged value; forward-fill per zone if missing
    route_df["weather_risk_enroute"] = route_df["weather_risk_enroute_new"].combine_first(
        route_df["weather_risk_enroute"]
    )
    route_df = route_df.drop(columns=["weather_risk_enroute_new", "weather_zone"], errors="ignore")

    # Forward-fill remaining NaNs
    route_df["weather_risk_enroute"] = route_df["weather_risk_enroute"].fillna(1.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    route_df.to_csv(out_path, index=False)
    print(f"Saved route feature matrix to {out_path}  ({route_df.shape[0]:,} rows x {route_df.shape[1]} cols)")

    return route_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build HEC fleet feature matrices")
    parser.add_argument("--route", action="store_true", help="Also build route feature matrix")
    args = parser.parse_args()

    df = build_fleet_feature_matrix()
    print(f"\nDemand features shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Nulls: {df.isnull().sum().sum()}")

    if args.route:
        rdf = build_route_feature_matrix()
        print(f"\nRoute features shape: {rdf.shape}")
        print(f"Route columns: {list(rdf.columns)}")
