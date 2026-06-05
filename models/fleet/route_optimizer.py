"""
route_optimizer.py
HEC AI Platform – Task 8: Model 2B – Greedy Fleet Route Optimizer

No ML training — rule-based scoring and greedy assignment.

Functions:
  build_optimizer              – precompute distance matrix & fleet specs, save config
  optimize_fleet_assignments   – greedy (vessel, port) assignment
  compute_fleet_kpis           – KPI summary for an assignment plan
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants – scoring weights
# ---------------------------------------------------------------------------
W_URGENCY    = 0.30
W_EFFICIENCY = 0.25
W_PROXIMITY  = 0.25
W_RISK       = 0.10
W_COST       = 0.10

AVG_SPEED_KNOTS = 12.0
SLA_DAYS = 14          # maximum days between collections

# Beaufort / wave-height thresholds per vessel class
# No-Go if max_wave_height_m >= threshold
NO_GO_WAVE_HEIGHT = {
    "Coastal Small":   3.0,
    "Coastal Medium":  4.0,
    "Regional":        5.5,
    "Offshore Large":  7.0,
}

# Fuel price fallback (USD/mt) if not provided in market df
FUEL_PRICE_FALLBACK = 500.0
EUR_USD_RATE = 1.08

# Revenue constants (matching fleet_features.py)
ACCEPTANCE_FEE_EUR_M3 = 30.0
RECOVERY_FRACTION = 0.30
RECOVERY_PRICE_EUR_M3 = 400.0
PORT_CHARGES_EUR = 1500.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _model_dir_path(model_dir: str = "models/trained") -> Path:
    p = Path(model_dir)
    if not p.is_absolute():
        p = _REPO_ROOT / model_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_distance_matrix() -> dict[tuple[str, str], float]:
    """Return a dict keyed by (origin, destination) -> distance_nm."""
    dist_path = _REPO_ROOT / "data/reference/port_distance_matrix.csv"
    if not dist_path.exists():
        return {}
    dist_df = pd.read_csv(dist_path)
    lookup: dict[tuple[str, str], float] = {}
    for _, row in dist_df.iterrows():
        lookup[(str(row["origin_port"]), str(row["destination_port"]))] = float(row["distance_nm"])
    return lookup


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    R_nm = 3440.065
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R_nm * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_optimizer(model_dir: str = "models/trained") -> dict:
    """
    Pre-compute distance matrix and fleet specs; save config to
    models/trained/route_optimizer_config.joblib.

    Returns
    -------
    dict with keys: distance_lookup (size), fleet_count, port_count
    """
    import joblib

    mdir = _model_dir_path(model_dir)

    distance_lookup = _load_distance_matrix()

    # Load fleet master
    fleet_path = _REPO_ROOT / "data/reference/fleet_master.csv"
    fleet_specs: dict = {}
    if fleet_path.exists():
        fm = pd.read_csv(fleet_path)
        for _, row in fm.iterrows():
            fleet_specs[row["vessel_id"]] = {
                "vessel_name":  row["vessel_name"],
                "vessel_class": row["vessel_class"],
                "tank_capacity_m3": float(row["tank_capacity_m3"]),
                "fuel_consumption_mt_day_laden":   float(row["fuel_consumption_mt_day_laden"]),
                "fuel_consumption_mt_day_ballast": float(row["fuel_consumption_mt_day_ballast"]),
                "max_speed_knots": float(row["max_speed_knots"]),
                "home_port": row["home_port"],
            }

    # Load port coords
    ports_path = _REPO_ROOT / "data/reference/ports.csv"
    port_coords: dict = {}
    if ports_path.exists():
        ports_df = pd.read_csv(ports_path)
        for _, row in ports_df.iterrows():
            port_coords[row["port_code"]] = (float(row["latitude"]), float(row["longitude"]))

    config = {
        "distance_lookup": distance_lookup,
        "fleet_specs": fleet_specs,
        "port_coords": port_coords,
    }

    config_path = mdir / "route_optimizer_config.joblib"
    # Safe: writing and loading a config dict we construct here from
    # reference CSVs we own — never loaded from untrusted sources.
    import joblib
    joblib.dump(config, config_path)
    print(f"Route optimizer config saved -> {config_path}")
    print(f"  Distance pairs: {len(distance_lookup):,}")
    print(f"  Fleet vessels:  {len(fleet_specs)}")
    print(f"  Port coords:    {len(port_coords)}")

    return {
        "distance_lookup_size": len(distance_lookup),
        "fleet_count": len(fleet_specs),
        "port_count": len(port_coords),
    }


def optimize_fleet_assignments(
    fleet_status: pd.DataFrame,
    port_demand: pd.DataFrame,
    weather: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Greedy (vessel, port) assignment using a scoring function.

    Parameters
    ----------
    fleet_status : pd.DataFrame
        Current vessel positions & status. Expected columns:
        vessel_id, vessel_name, vessel_class, latitude, longitude,
        status, tank_capacity_m3, current_cargo_m3, fuel_remaining_mt,
        fuel_consumption_mt_day, max_speed_knots  (optional: uses 12 kn)
    port_demand : pd.DataFrame
        Expected columns: port_code, current_storage_fill_pct, days_until_full,
        waste_volume_collected_m3, latitude (or port_latitude), longitude (or port_longitude)
    weather : pd.DataFrame
        Expected columns: zone_id (or port_code), significant_wave_height_m
        (or weather_risk_enroute). Used to check No-Go conditions.
    market : pd.DataFrame
        Expected columns: rotterdam_hfo_380_usd_mt (or brent_crude_usd_bbl).
        Used for fuel cost calculations.

    Returns
    -------
    pd.DataFrame with columns:
        vessel_id, vessel_name, assigned_port, distance_nm,
        eta_hours, expected_volume_m3, expected_revenue_eur,
        fuel_cost_eur, expected_margin_eur, urgency_score, score
    """
    import joblib

    # ------------------------------------------------------------------
    # Load config (distance lookup + specs)
    # ------------------------------------------------------------------
    config_path = _model_dir_path("models/trained") / "route_optimizer_config.joblib"
    if config_path.exists():
        # Safe: config contains only plain dicts (distance lookup, fleet specs,
        # port coords) built from our own reference CSVs in build_optimizer().
        # It is never received from or written by an untrusted external source.
        config = joblib.load(config_path)
        distance_lookup: dict = config.get("distance_lookup", {})
        fleet_specs: dict = config.get("fleet_specs", {})
        port_coords: dict = config.get("port_coords", {})
    else:
        distance_lookup = _load_distance_matrix()
        fleet_specs = {}
        port_coords = {}

    # ------------------------------------------------------------------
    # Fuel price
    # ------------------------------------------------------------------
    fuel_price_usd_mt = FUEL_PRICE_FALLBACK
    if "rotterdam_hfo_380_usd_mt" in market.columns and not market.empty:
        val = market["rotterdam_hfo_380_usd_mt"].iloc[-1]
        if pd.notna(val):
            fuel_price_usd_mt = float(val)
    elif "brent_crude_usd_bbl" in market.columns and not market.empty:
        val = market["brent_crude_usd_bbl"].iloc[-1]
        if pd.notna(val):
            fuel_price_usd_mt = float(val) * 6.0  # rough bbl->mt conversion

    eur_usd = EUR_USD_RATE
    if "eur_usd_rate" in market.columns and not market.empty:
        v = market["eur_usd_rate"].iloc[-1]
        if pd.notna(v) and v > 0:
            eur_usd = float(v)

    fuel_price_eur_mt = fuel_price_usd_mt / eur_usd

    # ------------------------------------------------------------------
    # Weather lookup: port_code -> max wave height
    # ------------------------------------------------------------------
    wave_by_port: dict[str, float] = {}
    if not weather.empty:
        wave_col = None
        for c in ("significant_wave_height_m", "weather_risk_enroute", "max_hs"):
            if c in weather.columns:
                wave_col = c
                break
        if wave_col:
            port_col = "port_code" if "port_code" in weather.columns else "zone_id"
            if port_col in weather.columns:
                wave_by_port = (
                    weather.groupby(port_col)[wave_col].max().to_dict()
                )

    # ------------------------------------------------------------------
    # Port demand table: index by port_code
    # ------------------------------------------------------------------
    demand = port_demand.copy()
    # Normalise lat/lon column names
    if "port_latitude" in demand.columns and "latitude" not in demand.columns:
        demand = demand.rename(columns={"port_latitude": "latitude", "port_longitude": "longitude"})

    demand = demand.set_index("port_code")

    # ------------------------------------------------------------------
    # Fleet state — work on a mutable copy
    # ------------------------------------------------------------------
    fleet = fleet_status.copy()

    # Track which ports have already been assigned (one vessel per port max)
    assigned_ports: set = set()
    assignments: list[dict] = {}  # vessel_id -> assignment record

    # Available vessels: not Under Maintenance + enough capacity
    MIN_CAPACITY_M3 = 50.0

    def _vessel_is_available(row: pd.Series) -> bool:
        if str(row.get("status", "")).lower().startswith("under maintenance"):
            return False
        cap = float(row.get("tank_capacity_m3", 0))
        cargo = float(row.get("current_cargo_m3", 0))
        remaining = cap - cargo
        if remaining < MIN_CAPACITY_M3:
            return False
        return True

    available_vessels = fleet[fleet.apply(_vessel_is_available, axis=1)].copy()

    # ------------------------------------------------------------------
    # Score all feasible (vessel, port) pairs
    # ------------------------------------------------------------------
    pair_scores: list[dict] = []

    for _, vessel in available_vessels.iterrows():
        vid = vessel["vessel_id"]
        vname = vessel.get("vessel_name", vid)
        vclass = vessel.get("vessel_class", "Coastal Medium")
        vlat = float(vessel.get("latitude", 0))
        vlon = float(vessel.get("longitude", 0))
        tank_cap = float(vessel.get("tank_capacity_m3", 1000))
        cur_cargo = float(vessel.get("current_cargo_m3", 0))
        avail_cap = tank_cap - cur_cargo
        fuel_remaining = float(vessel.get("fuel_remaining_mt", 100))
        fuel_rate = float(vessel.get("fuel_consumption_mt_day", 6))
        speed = float(vessel.get("max_speed_knots", AVG_SPEED_KNOTS))

        fuel_autonomy_days = fuel_remaining / max(fuel_rate, 0.1)
        no_go_wave = NO_GO_WAVE_HEIGHT.get(vclass, 4.0)

        spec = fleet_specs.get(vid, {})
        if spec:
            fuel_rate_laden = spec.get("fuel_consumption_mt_day_laden", fuel_rate)
        else:
            fuel_rate_laden = fuel_rate

        for port_code, d_row in demand.iterrows():
            # ----- Feasibility checks ---------------------------------
            # Weather No-Go
            wave_ht = wave_by_port.get(str(port_code), 0.0)
            if wave_ht >= no_go_wave:
                continue

            # Distance
            if port_code in port_coords:
                plat, plon = port_coords[str(port_code)]
            elif "latitude" in d_row and "longitude" in d_row:
                plat = float(d_row["latitude"])
                plon = float(d_row["longitude"])
            else:
                plat, plon = vlat, vlon

            dist_key = (None, str(port_code))
            # Try to find vessel's current port from destination_port or home_port
            cur_port = vessel.get("destination_port") or spec.get("home_port", "")
            if cur_port:
                dist_key = (str(cur_port), str(port_code))

            if dist_key in distance_lookup:
                distance_nm = distance_lookup[dist_key]
            else:
                distance_nm = _haversine_nm(vlat, vlon, plat, plon)

            # Fuel autonomy check: distance in days = distance_nm / (speed*24)
            travel_days = distance_nm / max(speed * 24.0, 1.0)
            if fuel_autonomy_days < travel_days:
                continue

            # ----- Scoring features -----------------------------------
            fill_pct = float(d_row.get("current_storage_fill_pct", 50))
            days_until_full = float(d_row.get("days_until_full", 7))
            waste_vol = float(d_row.get("waste_volume_collected_m3", 100))

            urgency_score = (fill_pct / 100.0) * (1.0 / max(days_until_full, 1.0))

            collection_efficiency = waste_vol / max(distance_nm, 10.0)

            proximity_score = 1.0 / max(distance_nm, 10.0)

            weather_risk = min(wave_ht / max(no_go_wave, 1.0), 1.0)

            # Fuel cost per m³
            fuel_for_trip = travel_days * fuel_rate_laden
            fuel_cost_eur = fuel_for_trip * fuel_price_eur_mt
            collected_vol = min(waste_vol, avail_cap)
            fuel_cost_per_m3 = fuel_cost_eur / max(collected_vol, 1.0)

            score = (
                W_URGENCY    * urgency_score
                + W_EFFICIENCY * min(collection_efficiency, 1.0)
                + W_PROXIMITY  * proximity_score * 1000   # scale: proximity is ~0.001-0.1
                - W_RISK       * weather_risk
                - W_COST       * min(fuel_cost_per_m3 / 100.0, 1.0)
            )

            # Revenue / margin
            revenue_eur = collected_vol * (ACCEPTANCE_FEE_EUR_M3 + RECOVERY_FRACTION * RECOVERY_PRICE_EUR_M3)
            margin_eur = revenue_eur - fuel_cost_eur - PORT_CHARGES_EUR
            eta_hours = travel_days * 24.0

            pair_scores.append({
                "vessel_id":            vid,
                "vessel_name":          vname,
                "vessel_class":         vclass,
                "port_code":            str(port_code),
                "distance_nm":          round(distance_nm, 1),
                "eta_hours":            round(eta_hours, 1),
                "expected_volume_m3":   round(collected_vol, 1),
                "expected_revenue_eur": round(revenue_eur, 0),
                "fuel_cost_eur":        round(fuel_cost_eur, 0),
                "expected_margin_eur":  round(margin_eur, 0),
                "urgency_score":        round(urgency_score, 4),
                "score":                round(score, 6),
            })

    if not pair_scores:
        return pd.DataFrame(columns=[
            "vessel_id", "vessel_name", "assigned_port", "distance_nm",
            "eta_hours", "expected_volume_m3", "expected_revenue_eur",
            "fuel_cost_eur", "expected_margin_eur", "urgency_score", "score",
        ])

    scores_df = pd.DataFrame(pair_scores).sort_values("score", ascending=False)

    # ------------------------------------------------------------------
    # Greedy assignment: best score first; each vessel and port used once
    # ------------------------------------------------------------------
    assigned_vessels: set = set()
    result_rows: list[dict] = []

    for _, row in scores_df.iterrows():
        vid = row["vessel_id"]
        port = row["port_code"]
        if vid in assigned_vessels:
            continue
        if port in assigned_ports:
            continue
        assigned_vessels.add(vid)
        assigned_ports.add(port)
        result_rows.append({
            "vessel_id":            vid,
            "vessel_name":          row["vessel_name"],
            "assigned_port":        port,
            "distance_nm":          row["distance_nm"],
            "eta_hours":            row["eta_hours"],
            "expected_volume_m3":   row["expected_volume_m3"],
            "expected_revenue_eur": row["expected_revenue_eur"],
            "fuel_cost_eur":        row["fuel_cost_eur"],
            "expected_margin_eur":  row["expected_margin_eur"],
            "urgency_score":        row["urgency_score"],
            "score":                row["score"],
        })

    return pd.DataFrame(result_rows).reset_index(drop=True)


def compute_fleet_kpis(
    assignments: pd.DataFrame,
    fleet_status: pd.DataFrame,
) -> dict:
    """
    Compute fleet-level KPIs from an assignment plan.

    Returns
    -------
    dict with keys:
        fleet_utilization_pct        – assigned vessels / total available
        avg_empty_miles_pct          – placeholder (0 if no ballast data)
        sla_compliance_pct           – ports assigned before 14-day SLA / total ports with demand
        total_expected_revenue        – sum of expected revenues
        total_fuel_cost               – sum of fuel costs
    """
    if assignments.empty:
        return {
            "fleet_utilization_pct": 0.0,
            "avg_empty_miles_pct": 0.0,
            "sla_compliance_pct": 0.0,
            "total_expected_revenue": 0.0,
            "total_fuel_cost": 0.0,
        }

    # Available vessels (not under maintenance, sufficient capacity)
    def _is_available(row: pd.Series) -> bool:
        if str(row.get("status", "")).lower().startswith("under maintenance"):
            return False
        cap = float(row.get("tank_capacity_m3", 0))
        cargo = float(row.get("current_cargo_m3", 0))
        return (cap - cargo) >= 50.0

    available_count = fleet_status.apply(_is_available, axis=1).sum()
    assigned_count = assignments["vessel_id"].nunique()

    fleet_utilization_pct = (
        round(assigned_count / max(available_count, 1) * 100, 1)
    )

    # Empty-miles: without actual ballast leg data we report 0
    avg_empty_miles_pct = 0.0

    # SLA compliance: assignments where ETA < SLA_DAYS * 24 hours
    if "eta_hours" in assignments.columns:
        compliant = (assignments["eta_hours"] <= SLA_DAYS * 24).sum()
        sla_compliance_pct = round(compliant / max(len(assignments), 1) * 100, 1)
    else:
        sla_compliance_pct = 0.0

    total_expected_revenue = float(assignments["expected_revenue_eur"].sum())
    total_fuel_cost = float(assignments["fuel_cost_eur"].sum())

    return {
        "fleet_utilization_pct": fleet_utilization_pct,
        "avg_empty_miles_pct": avg_empty_miles_pct,
        "sla_compliance_pct": sla_compliance_pct,
        "total_expected_revenue": round(total_expected_revenue, 0),
        "total_fuel_cost": round(total_fuel_cost, 0),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = build_optimizer()
    print(f"\nOptimizer built: {result}")

    # Quick smoke-test with latest data
    fleet_df  = pd.read_csv(_REPO_ROOT / "data/raw/fleet_status.csv")
    demand_df = pd.read_csv(_REPO_ROOT / "data/raw/port_waste_demand.csv")
    weather_df = pd.read_csv(_REPO_ROOT / "data/raw/maritime_weather.csv")
    market_df  = pd.read_csv(_REPO_ROOT / "data/raw/oil_market.csv")

    # Use latest snapshot
    fleet_df["timestamp"] = pd.to_datetime(fleet_df["timestamp"])
    latest_date = fleet_df["timestamp"].max().normalize()
    fleet_snap = fleet_df[fleet_df["timestamp"].dt.normalize() == latest_date].copy()
    fleet_snap = fleet_snap.groupby("vessel_id").last().reset_index()

    demand_df["date"] = pd.to_datetime(demand_df["date"])
    demand_snap = demand_df[demand_df["date"] == demand_df["date"].max()].copy()

    assignments = optimize_fleet_assignments(fleet_snap, demand_snap, weather_df, market_df)
    print(f"\nAssignments ({len(assignments)} vessels):")
    print(assignments[["vessel_id", "assigned_port", "distance_nm", "score"]].to_string(index=False))

    kpis = compute_fleet_kpis(assignments, fleet_snap)
    print(f"\nKPIs: {kpis}")
