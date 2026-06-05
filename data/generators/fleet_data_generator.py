"""
fleet_data_generator.py
Generates synthetic operational data for the HEC AI Platform:
  1. Port waste demand records
  2. HEC fleet position / status snapshots
  3. Voyage history
  4. Offshore platform daily data
  5. Maritime weather by zone
  6. Oil / fuel market prices
  7. Orchestrator: generate_all()
"""

import math
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup – allow running as a standalone script
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.generators.common import (
    PORTS,
    seasonal_factor,
    haversine_distance_nm,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------------------
# Reference data paths
# ---------------------------------------------------------------------------
_REF = _REPO_ROOT / "data" / "reference"

# ---------------------------------------------------------------------------
# Helper: load reference CSVs once (module-level cache)
# ---------------------------------------------------------------------------

def _load_ports() -> pd.DataFrame:
    return pd.read_csv(_REF / "ports.csv")


def _load_fleet() -> pd.DataFrame:
    return pd.read_csv(_REF / "fleet_master.csv")


def _load_distance_matrix() -> pd.DataFrame:
    return pd.read_csv(_REF / "port_distance_matrix.csv")


def _load_platforms() -> pd.DataFrame:
    return pd.read_csv(_REF / "offshore_platforms.csv")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _date_range(start: date, end: date) -> list:
    """Return list of date objects from start (inclusive) to end (exclusive)."""
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days)]


def _is_med_port(port_code: str) -> bool:
    """True for Mediterranean / southern European ports."""
    non_med = {"HAM", "RTM"}
    return port_code not in non_med


def _is_north_eu(port_code: str) -> bool:
    return port_code in {"HAM", "RTM"}


def _year_growth(d: date, base_year: int = 2022, rate: float = 0.03) -> float:
    """Compound annual growth multiplier relative to base_year."""
    years = (d - date(base_year, 1, 1)).days / 365.25
    return (1 + rate) ** years


# ============================================================
# Function 1 – Port Waste Demand
# ============================================================

def generate_port_waste_demand(n_days: int = 1612) -> pd.DataFrame:
    """
    ~22,568 rows: 14 ports × 1612 days (2022-01-01 to 2026-06-01).
    Note: the spec figure of ~80,000 was an overestimate; the actual
    count is 14 × 1612 ≈ 22,568.
    """
    rng = np.random.default_rng(42)

    ports_df = _load_ports()
    start = date(2022, 1, 1)
    end = date(2026, 6, 1)
    all_dates = _date_range(start, end)[:n_days]

    records = []

    for _, port_row in ports_df.iterrows():
        pcode = port_row["port_code"]
        pname = port_row["port_name"]
        pcountry = port_row["country"]
        plat = port_row["latitude"]
        plon = port_row["longitude"]
        base_calls = port_row["avg_daily_vessel_calls"]
        hec_share_base = port_row["hec_market_share_pct"]
        storage_cap = port_row["storage_capacity_m3"]
        competitor_count = int(port_row["competitor_count"])
        competitor_presence = competitor_count > 0

        # Storage simulation state
        fill_pct = rng.uniform(20, 50)
        # Busy port = smaller interval, quiet port = larger interval
        if base_calls >= 80:
            collection_interval = rng.integers(3, 8)
        elif base_calls >= 40:
            collection_interval = rng.integers(5, 15)
        else:
            collection_interval = rng.integers(10, 21)
        days_since_collection = 0

        # Monthly market share variation state (changes monthly)
        current_month = None
        monthly_share_offset = 0.0

        for d in all_dates:
            # Monthly variation in HEC share
            if d.month != current_month:
                current_month = d.month
                monthly_share_offset = rng.uniform(-3, 3)

            sf = seasonal_factor(d, pcode)
            growth = _year_growth(d)

            # Weekday effects
            dow = d.weekday()  # 0=Mon … 6=Sun
            if dow == 5:
                weekend_mult = 0.85
            elif dow == 6:
                weekend_mult = 0.75
            else:
                weekend_mult = 1.0

            effective_calls = base_calls * sf * growth * weekend_mult

            # Vessel type split: rough proportions, vary slightly per port
            if pcode in {"PIR", "MLT", "LIM"}:
                # More tanker / cruise activity
                container_frac = rng.uniform(0.28, 0.35)
                tanker_frac    = rng.uniform(0.25, 0.35)
                bulk_frac      = rng.uniform(0.15, 0.22)
                cruise_base    = 0.08
                other_frac     = 1.0 - container_frac - tanker_frac - bulk_frac - cruise_base
            elif pcode in {"RTM", "HAM", "BCN", "GEN"}:
                container_frac = rng.uniform(0.38, 0.50)
                tanker_frac    = rng.uniform(0.18, 0.28)
                bulk_frac      = rng.uniform(0.18, 0.25)
                cruise_base    = 0.03
                other_frac     = 1.0 - container_frac - tanker_frac - bulk_frac - cruise_base
            elif pcode in {"IST", "PSD", "ALE"}:
                container_frac = rng.uniform(0.30, 0.40)
                tanker_frac    = rng.uniform(0.25, 0.35)
                bulk_frac      = rng.uniform(0.20, 0.28)
                cruise_base    = 0.03
                other_frac     = 1.0 - container_frac - tanker_frac - bulk_frac - cruise_base
            else:  # GIB, TAN, ALG, MRS
                container_frac = rng.uniform(0.35, 0.45)
                tanker_frac    = rng.uniform(0.28, 0.38)
                bulk_frac      = rng.uniform(0.12, 0.20)
                cruise_base    = 0.04
                other_frac     = 1.0 - container_frac - tanker_frac - bulk_frac - cruise_base
            other_frac = max(0.02, other_frac)

            # Cruise ships: 0 in Dec-Mar for Med ports; peak Jul-Sep
            if _is_med_port(pcode):
                if d.month in (12, 1, 2, 3):
                    cruise_day_factor = 0.0
                elif d.month in (7, 8, 9):
                    cruise_day_factor = rng.uniform(0.8, 1.5)
                else:
                    cruise_day_factor = rng.uniform(0.3, 0.8)
            else:
                cruise_day_factor = rng.uniform(0.1, 0.3)

            total_non_cruise = effective_calls * (1 - cruise_base)
            n_container = max(0.0, total_non_cruise * container_frac + rng.normal(0, 1))
            n_tanker    = max(0.0, total_non_cruise * tanker_frac + rng.normal(0, 0.8))
            n_bulk      = max(0.0, total_non_cruise * bulk_frac + rng.normal(0, 0.8))
            n_other     = max(0.0, total_non_cruise * other_frac + rng.normal(0, 0.5))
            n_cruise    = max(0.0, effective_calls * cruise_base * cruise_day_factor + rng.normal(0, 0.3))

            # Cap cruise at 15/day as per spec
            n_cruise = min(n_cruise, 15.0)
            n_total  = n_container + n_tanker + n_bulk + n_cruise + n_other

            # Waste volumes (m³/call base rates) with ±20% noise
            def noisy(val):
                return max(0.0, val * rng.uniform(0.80, 1.20))

            bilge  = noisy(n_container * 5 + n_tanker * 3 + n_bulk * 4 + n_cruise * 15 + n_other * 3)
            sludge = noisy(n_container * 3 + n_tanker * 8 + n_bulk * 5 + n_cruise * 8  + n_other * 2)
            slops  = noisy(n_tanker * 20 + n_other * 2)
            waste_other = noisy(n_cruise * 5 + n_other * 1)
            total_waste = bilge + sludge + slops + waste_other

            # Avg vessel size (GT) – roughly correlates with traffic type
            avg_gt = (n_container * 60_000 + n_tanker * 45_000 + n_bulk * 55_000 +
                      n_cruise * 120_000 + n_other * 8_000) / max(n_total, 1)
            avg_gt = avg_gt * rng.uniform(0.9, 1.1)

            # Market share
            hec_share = min(100.0, max(0.0, hec_share_base + monthly_share_offset))

            # Storage fill simulation
            fill_rate_per_day = max(0.1, total_waste / storage_cap * 100 * rng.uniform(0.7, 1.3))
            days_since_collection += 1
            fill_pct += fill_rate_per_day

            # Collection trigger
            if days_since_collection >= collection_interval or fill_pct >= 85:
                fill_pct = rng.uniform(15, 30)
                days_since_collection = 0
                if base_calls >= 80:
                    collection_interval = rng.integers(3, 8)
                elif base_calls >= 40:
                    collection_interval = rng.integers(5, 15)
                else:
                    collection_interval = rng.integers(10, 21)

            fill_pct = min(fill_pct, 98.0)
            days_until_full = max(0.0, (100.0 - fill_pct) / max(fill_rate_per_day, 0.1))

            records.append({
                "demand_id": f"DEM-{pcode}-{d.strftime('%Y%m%d')}",
                "date": d.isoformat(),
                "port_code": pcode,
                "port_name": pname,
                "port_country": pcountry,
                "port_latitude": plat,
                "port_longitude": plon,
                "vessel_calls_total": round(n_total, 1),
                "vessel_calls_by_type_container": round(n_container, 1),
                "vessel_calls_by_type_tanker": round(n_tanker, 1),
                "vessel_calls_by_type_bulk": round(n_bulk, 1),
                "vessel_calls_by_type_cruise": round(n_cruise, 1),
                "vessel_calls_by_type_other": round(n_other, 1),
                "waste_volume_collected_m3": round(total_waste, 1),
                "waste_bilge_m3": round(bilge, 1),
                "waste_sludge_m3": round(sludge, 1),
                "waste_slops_m3": round(slops, 1),
                "waste_other_m3": round(waste_other, 1),
                "avg_vessel_size_gt": round(avg_gt),
                "hec_market_share_pct": round(hec_share, 1),
                "competitor_presence": competitor_presence,
                "port_reception_facility_capacity_m3": storage_cap,
                "current_storage_fill_pct": round(fill_pct, 1),
                "days_until_full": round(days_until_full, 1),
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ============================================================
# Function 2 – HEC Fleet Status Snapshots
# ============================================================

def generate_hec_fleet_status(n_snapshots_per_vessel: int = 2000) -> pd.DataFrame:
    """
    ~50,000 rows: 25 vessels × 2000 snapshots (~every 20 hours, 4.5 years).
    """
    rng = np.random.default_rng(1234)

    fleet_df = _load_fleet()
    ports_df = _load_ports()

    port_coords: Dict[str, tuple] = {
        row["port_code"]: (row["latitude"], row["longitude"])
        for _, row in ports_df.iterrows()
    }

    # Full date range
    start_ts = datetime(2022, 1, 1, 0, 0)
    end_ts   = datetime(2026, 6, 1, 0, 0)
    total_hrs = (end_ts - start_ts).total_seconds() / 3600.0
    interval_hrs = total_hrs / max(n_snapshots_per_vessel, 1)

    # Pre-select vetting status for each vessel (80/15/5)
    vetting_options = ["Approved", "Conditional", "Expired"]
    vetting_probs   = [0.80, 0.15, 0.05]

    all_port_codes = list(port_coords.keys())
    records = []

    for _, vessel in fleet_df.iterrows():
        vid   = vessel["vessel_id"]
        vname = vessel["vessel_name"]
        vclass = vessel["vessel_class"]
        dwt   = vessel["dwt"]
        tank_cap = vessel["tank_capacity_m3"]
        home  = vessel["home_port"]
        fuel_laden  = vessel["fuel_consumption_mt_day_laden"]
        fuel_ballast = vessel["fuel_consumption_mt_day_ballast"]
        max_spd = vessel["max_speed_knots"]
        ice_class = vessel["ice_class"]

        # Pick vetting
        vetting = str(rng.choice(vetting_options, p=vetting_probs))

        # Last inspection date: sometime in 2021 or early 2022
        insp_offset_days = int(rng.integers(0, 400))
        last_insp = (date(2021, 1, 1) + timedelta(days=insp_offset_days)).isoformat()

        # Maintenance: 14 days/year → pick random start day within first year
        maint_month = int(rng.integers(1, 13))
        maint_day   = int(rng.integers(1, 28))
        maint_start = datetime(2022, maint_month, maint_day, 0, 0)

        # Voyage planning by class
        if "Coastal Small" in vclass:
            port_pool = [p for p in all_port_codes if _is_med_port(p) or p == home]
            avg_dist = 25.0
            avg_spd  = max_spd * 0.80
            port_time_hrs_range = (4, 12)
            max_cargo_load = tank_cap * 0.70
        elif "Coastal Medium" in vclass:
            port_pool = all_port_codes
            avg_dist = 200.0
            avg_spd  = max_spd * 0.82
            port_time_hrs_range = (8, 18)
            max_cargo_load = tank_cap * 0.80
        elif "Regional" in vclass:
            port_pool = all_port_codes
            avg_dist = 300.0
            avg_spd  = max_spd * 0.85
            port_time_hrs_range = (12, 24)
            max_cargo_load = tank_cap * 0.85
        else:  # Offshore Large
            port_pool = all_port_codes
            avg_dist = 400.0
            avg_spd  = max_spd * 0.88
            port_time_hrs_range = (18, 36)
            max_cargo_load = tank_cap * 0.90

        # Initial state
        cur_lat, cur_lon = port_coords.get(home, (37.9, 23.6))
        cur_status  = "At Port Loading"
        cur_cargo   = 0.0
        cargo_type  = "Oily Waste Mix"
        dest_port   = home
        fuel_cap    = dwt * 0.05  # rough fuel tank capacity (mt)
        fuel_remain = fuel_cap * rng.uniform(0.80, 0.95)
        crew_hrs    = rng.uniform(400, 720)
        days_maint  = int(rng.integers(30, 180))

        # Voyage tracking state
        origin_lat, origin_lon = cur_lat, cur_lon
        dest_lat, dest_lon = cur_lat, cur_lon
        voyage_progress = 0.0   # 0.0 → 1.0
        voyage_dist_nm  = 0.0
        voyage_hrs_total = 0.0
        in_port_remaining_hrs = float(rng.integers(*port_time_hrs_range))
        # 0=loading at port, 1=laden transit, 2=discharging at port, 3=ballast transit
        phase = 0

        # Annual maintenance: build a list of maintenance windows (one per year)
        maint_windows = []
        for yr in range(2022, 2027):
            mo = int(rng.integers(1, 13))
            dy = int(rng.integers(1, 28))
            try:
                ms = datetime(yr, mo, dy, 0, 0)
            except ValueError:
                ms = datetime(yr, mo, 1, 0, 0)
            maint_windows.append((ms, ms + timedelta(days=14)))

        for snap_i in range(n_snapshots_per_vessel):
            ts = start_ts + timedelta(hours=snap_i * interval_hrs)

            # Check maintenance window
            in_maintenance = any(ws <= ts <= we for ws, we in maint_windows)
            if in_maintenance:
                cur_status = "Under Maintenance"
                cur_lat, cur_lon = port_coords.get(home, (37.9, 23.6))
                speed = 0.0
                heading = 0.0
                dest_port = home
                eta_dest  = ts + timedelta(days=3)
                cur_cargo = 0.0
            else:
                # Advance phase by interval_hrs
                dt_hrs = interval_hrs

                if phase == 0:  # At Port Loading
                    cur_status = "At Port Loading"
                    speed = 0.0
                    heading = 0.0
                    dest_lat2, dest_lon2 = port_coords.get(dest_port, (cur_lat, cur_lon))
                    eta_dest = ts + timedelta(hours=max(1, in_port_remaining_hrs))
                    in_port_remaining_hrs -= dt_hrs

                    # Loading cargo
                    load_rate = max_cargo_load / max(in_port_remaining_hrs + dt_hrs, 4)
                    cur_cargo = min(cur_cargo + load_rate * dt_hrs, max_cargo_load)

                    # Fuel: hotel load ~0.3× ballast rate
                    fuel_remain -= (fuel_ballast * 0.3) * dt_hrs / 24
                    crew_hrs    -= dt_hrs

                    if in_port_remaining_hrs <= 0:
                        # Choose destination for laden transit
                        if "Coastal Small" in vclass:
                            # pick nearby port
                            near_ports = [p for p in port_pool if p != dest_port]
                            dest_port = str(rng.choice(near_ports))
                        elif "Offshore Large" in vclass:
                            # discharge at home / treatment facility
                            dest_port = home
                        else:
                            candidates = [p for p in all_port_codes if p != dest_port]
                            dest_port = str(rng.choice(candidates))

                        dest_lat2, dest_lon2 = port_coords.get(dest_port, (cur_lat, cur_lon))
                        dist = haversine_distance_nm(cur_lat, cur_lon, dest_lat2, dest_lon2)
                        voyage_dist_nm  = max(dist, 5.0)
                        voyage_hrs_total = voyage_dist_nm / max(avg_spd, 1.0)
                        voyage_progress  = 0.0
                        origin_lat, origin_lon = cur_lat, cur_lon
                        phase = 1
                        in_port_remaining_hrs = float(rng.integers(*port_time_hrs_range))

                elif phase == 1:  # In Transit Laden
                    cur_status = "In Transit Laden"
                    voyage_progress += dt_hrs / max(voyage_hrs_total, 0.1)
                    voyage_progress = min(voyage_progress, 1.0)

                    dest_lat2, dest_lon2 = port_coords.get(dest_port, (cur_lat, cur_lon))
                    cur_lat = origin_lat + (dest_lat2 - origin_lat) * voyage_progress
                    cur_lon = origin_lon + (dest_lon2 - origin_lon) * voyage_progress
                    speed = avg_spd * rng.uniform(0.90, 1.05)

                    # Heading toward destination
                    dlat = dest_lat2 - cur_lat
                    dlon = dest_lon2 - cur_lon
                    heading = math.degrees(math.atan2(dlon, dlat)) % 360

                    eta_dest = ts + timedelta(hours=max(0.5, voyage_hrs_total * (1 - voyage_progress)))
                    fuel_remain -= fuel_laden * dt_hrs / 24
                    crew_hrs    -= dt_hrs

                    if voyage_progress >= 1.0:
                        cur_lat, cur_lon = dest_lat2, dest_lon2
                        phase = 2
                        voyage_progress = 0.0

                elif phase == 2:  # At Port Discharging
                    cur_status = "At Port Discharging"
                    speed = 0.0
                    heading = 0.0
                    dest_lat2, dest_lon2 = port_coords.get(dest_port, (cur_lat, cur_lon))
                    eta_dest = ts + timedelta(hours=max(1, in_port_remaining_hrs))
                    in_port_remaining_hrs -= dt_hrs

                    # Discharge cargo
                    discharge_rate = cur_cargo / max(in_port_remaining_hrs + dt_hrs, 4)
                    cur_cargo = max(cur_cargo - discharge_rate * dt_hrs, 0.0)

                    # Refuel and crew reset at home port
                    if dest_port == home:
                        fuel_remain = fuel_cap * rng.uniform(0.80, 0.95)
                        crew_hrs    = 720.0
                    else:
                        fuel_remain -= (fuel_ballast * 0.3) * dt_hrs / 24
                        crew_hrs    -= dt_hrs

                    if in_port_remaining_hrs <= 0:
                        # Choose next collection port for ballast transit
                        candidates = [p for p in port_pool if p != dest_port]
                        next_port = str(rng.choice(candidates))
                        origin_lat, origin_lon = cur_lat, cur_lon
                        dest_lat2, dest_lon2 = port_coords.get(next_port, (cur_lat, cur_lon))
                        dist = haversine_distance_nm(cur_lat, cur_lon, dest_lat2, dest_lon2)
                        voyage_dist_nm  = max(dist, 5.0)
                        voyage_hrs_total = voyage_dist_nm / max(avg_spd, 1.0)
                        voyage_progress  = 0.0
                        dest_port = next_port
                        phase = 3
                        in_port_remaining_hrs = float(rng.integers(*port_time_hrs_range))

                else:  # phase == 3: In Transit Ballast
                    cur_status = "In Transit Ballast"
                    voyage_progress += dt_hrs / max(voyage_hrs_total, 0.1)
                    voyage_progress = min(voyage_progress, 1.0)

                    dest_lat2, dest_lon2 = port_coords.get(dest_port, (cur_lat, cur_lon))
                    cur_lat = origin_lat + (dest_lat2 - origin_lat) * voyage_progress
                    cur_lon = origin_lon + (dest_lon2 - origin_lon) * voyage_progress
                    speed = avg_spd * rng.uniform(0.88, 1.02)

                    dlat = dest_lat2 - cur_lat
                    dlon = dest_lon2 - cur_lon
                    heading = math.degrees(math.atan2(dlon, dlat)) % 360

                    eta_dest = ts + timedelta(hours=max(0.5, voyage_hrs_total * (1 - voyage_progress)))
                    fuel_remain -= fuel_ballast * dt_hrs / 24
                    crew_hrs    -= dt_hrs

                    if voyage_progress >= 1.0:
                        cur_lat, cur_lon = dest_lat2, dest_lon2
                        phase = 0
                        voyage_progress = 0.0

                # Clamp fuel and crew
                fuel_remain = max(fuel_remain, 0.0)
                crew_hrs    = max(crew_hrs, 0.0)
                days_maint  = max(0, days_maint - dt_hrs / 24)

            # Recalculate maintenance countdown
            days_maint = max(0.0, days_maint - interval_hrs / 24)
            if in_maintenance:
                days_maint = float(rng.integers(60, 365))

            cargo_pct = cur_cargo / tank_cap * 100 if tank_cap > 0 else 0.0

            records.append({
                "vessel_id":                   vid,
                "vessel_name":                 vname,
                "vessel_class":                vclass,
                "dwt":                         dwt,
                "tank_capacity_m3":            tank_cap,
                "home_port":                   home,
                "timestamp":                   ts.isoformat(),
                "latitude":                    round(cur_lat, 4),
                "longitude":                   round(cur_lon, 4),
                "speed_knots":                 round(speed, 1),
                "heading_deg":                 round(heading % 360, 1),
                "status":                      cur_status,
                "current_cargo_m3":            round(cur_cargo, 1),
                "current_cargo_pct":           round(cargo_pct, 1),
                "cargo_type_primary":          cargo_type,
                "destination_port":            dest_port,
                "eta_destination":             eta_dest.isoformat() if isinstance(eta_dest, datetime) else str(eta_dest),
                "fuel_remaining_mt":           round(fuel_remain, 1),
                "fuel_consumption_mt_day":     fuel_laden if "Laden" in cur_status else fuel_ballast,
                "days_until_next_maintenance": round(days_maint, 1),
                "crew_hours_remaining":        round(crew_hrs, 1),
                "ice_class":                   ice_class,
                "last_inspection_date":        last_insp,
                "vetting_status":              vetting,
            })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df


# ============================================================
# Function 3 – Voyage History
# ============================================================

def generate_voyage_history(
    fleet_status_df: Optional[pd.DataFrame] = None,
    demand_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    ~8,000 rows: 25 vessels × ~80 voyages/year × 4 years.
    """
    rng = np.random.default_rng(9999)

    fleet_df = _load_fleet()
    dist_df  = _load_distance_matrix()

    # Build distance lookup
    dist_lookup: Dict[tuple, float] = {}
    for _, row in dist_df.iterrows():
        dist_lookup[(row["origin_port"], row["destination_port"])] = float(row["distance_nm"])

    def get_dist(o, d):
        return dist_lookup.get((o, d), haversine_distance_nm(
            PORTS[o]["lat"], PORTS[o]["lon"],
            PORTS[d]["lat"], PORTS[d]["lon"]
        ) if o in PORTS and d in PORTS else 200.0)

    all_port_codes = list(PORTS.keys())

    start = datetime(2022, 1, 1)
    end   = datetime(2026, 6, 1)

    records = []
    voyage_seq = 1

    for _, vessel in fleet_df.iterrows():
        vid   = vessel["vessel_id"]
        vname = vessel["vessel_name"]
        vclass = vessel["vessel_class"]
        dwt   = vessel["dwt"]
        tank_cap = vessel["tank_capacity_m3"]
        home  = vessel["home_port"]
        fuel_rate_laden  = vessel["fuel_consumption_mt_day_laden"]
        fuel_rate_ballast = vessel["fuel_consumption_mt_day_ballast"]
        max_spd = vessel["max_speed_knots"]

        # Voyages per year and parameters by class
        if "Coastal Small" in vclass:
            voyages_per_year = int(rng.integers(150, 200))
            dist_range = (5, 50)
            collections_range = (3, 9)
            port_time_range   = (4, 12)
            rev_range  = (500, 20_000)
            port_pool  = [p for p in all_port_codes if _is_med_port(p)]
            avg_spd    = max_spd * 0.80
        elif "Coastal Medium" in vclass:
            voyages_per_year = int(rng.integers(80, 121))
            dist_range = (30, 400)
            collections_range = (1, 4)
            port_time_range   = (8, 18)
            rev_range  = (5_000, 100_000)
            port_pool  = all_port_codes
            avg_spd    = max_spd * 0.82
        elif "Regional" in vclass:
            voyages_per_year = int(rng.integers(40, 80))
            dist_range = (100, 500)
            collections_range = (1, 4)
            port_time_range   = (12, 24)
            rev_range  = (10_000, 150_000)
            port_pool  = all_port_codes
            avg_spd    = max_spd * 0.85
        else:  # Offshore Large
            voyages_per_year = int(rng.integers(20, 41))
            dist_range = (200, 500)
            collections_range = (1, 2)
            port_time_range   = (18, 36)
            rev_range  = (50_000, 500_000)
            port_pool  = all_port_codes
            avg_spd    = max_spd * 0.88

        total_days = (end - start).days
        voyage_interval_days = 365.0 / voyages_per_year

        cur_time = start + timedelta(hours=float(rng.uniform(0, 48)))
        cur_port = home

        while cur_time < end:
            # Pick departure and arrival ports
            dep_port = cur_port
            dest_candidates = [p for p in port_pool if p != dep_port]
            if not dest_candidates:
                dest_candidates = all_port_codes
            arr_port = str(rng.choice(dest_candidates))

            dist_nm = get_dist(dep_port, arr_port)
            # Constrain distance to class range (pick another port if too far / close)
            if not (dist_range[0] <= dist_nm <= dist_range[1] * 3):
                # Try to find a better match
                for _ in range(5):
                    candidate = str(rng.choice(dest_candidates))
                    d_try = get_dist(dep_port, candidate)
                    if dist_range[0] <= d_try <= dist_range[1] * 3:
                        arr_port = candidate
                        dist_nm = d_try
                        break

            year = cur_time.year
            month = cur_time.month
            is_winter_ns = month in (11, 12, 1, 2, 3)
            is_north_sea_route = dep_port in {"HAM", "RTM"} or arr_port in {"HAM", "RTM"}

            # Sea time
            actual_spd = avg_spd * rng.uniform(0.88, 1.02)
            sea_time = dist_nm / max(actual_spd, 0.5)

            # Weather delays
            if is_north_sea_route and is_winter_ns:
                weather_delay = float(rng.exponential(8.0))
            elif is_north_sea_route:
                weather_delay = float(rng.exponential(3.0))
            elif is_winter_ns:
                weather_delay = float(rng.exponential(3.0))
            else:
                weather_delay = float(rng.exponential(2.0))
            weather_delay = round(min(weather_delay, 72.0), 1)

            # Mechanical delays (90% = 0)
            if rng.random() < 0.10:
                mech_delay = float(rng.exponential(4.0))
            else:
                mech_delay = 0.0
            mech_delay = round(min(mech_delay, 24.0), 1)

            # Port times
            port_time = float(rng.integers(*port_time_range))
            # Waiting time: 0-48h, more in winter and congested ports
            if dep_port in {"RTM", "PIR", "IST", "HAM"} or is_winter_ns:
                waiting_time = float(rng.exponential(8.0))
            else:
                waiting_time = float(rng.exponential(3.0))
            waiting_time = round(min(waiting_time, 48.0), 1)

            voyage_duration = sea_time + port_time + weather_delay + mech_delay + waiting_time

            dep_time = cur_time
            arr_time = cur_time + timedelta(hours=voyage_duration)

            if arr_time >= end:
                break

            # Cargo
            collections_count = int(rng.integers(*collections_range))
            cargo_loaded = tank_cap * rng.uniform(0.50, 0.90)
            cargo_discharged = cargo_loaded * rng.uniform(0.95, 1.00)

            # Fuel
            fuel_consumed = (
                (sea_time / 24) * fuel_rate_laden
                + (port_time / 24) * 0.3 * fuel_rate_laden
            ) * rng.uniform(0.92, 1.08)

            # Fuel cost: Rotterdam HFO price ~400-600 $/mt → convert to EUR
            hfo_usd = rng.uniform(400, 600)
            eur_usd = rng.uniform(0.98, 1.12)
            fuel_cost_eur = fuel_consumed * (hfo_usd / eur_usd)

            # Port charges
            if dwt < 2000:
                port_charge_per_call = rng.uniform(200, 800)
            elif dwt < 10000:
                port_charge_per_call = rng.uniform(500, 1500)
            else:
                port_charge_per_call = rng.uniform(1000, 2000)
            port_charges_total = port_charge_per_call * (collections_count + 1)

            # Revenue: acceptance fees €15-50/m³
            acceptance_fee = rng.uniform(15, 50)
            revenue = cargo_loaded * acceptance_fee
            # Clamp to realistic ranges
            revenue = float(np.clip(revenue, rev_range[0], rev_range[1]))

            total_cost = fuel_cost_eur + port_charges_total
            voyage_margin = revenue - total_cost

            avg_collect_time = port_time / max(collections_count, 1)

            voy_year = dep_time.year
            voy_id = f"VOY-{voy_year}-{voyage_seq:05d}"
            voyage_seq += 1

            records.append({
                "voyage_id":              voy_id,
                "vessel_id":              vid,
                "vessel_name":            vname,
                "vessel_class":           vclass,
                "departure_port":         dep_port,
                "departure_time":         dep_time.isoformat(),
                "arrival_port":           arr_port,
                "arrival_time":           arr_time.isoformat(),
                "distance_nm":            round(dist_nm, 1),
                "voyage_duration_hrs":    round(voyage_duration, 1),
                "sea_time_hrs":           round(sea_time, 1),
                "port_time_hrs":          round(port_time, 1),
                "waiting_time_hrs":       round(waiting_time, 1),
                "avg_speed_knots":        round(actual_spd, 1),
                "fuel_consumed_mt":       round(fuel_consumed, 1),
                "cargo_loaded_m3":        round(cargo_loaded, 1),
                "cargo_discharged_m3":    round(cargo_discharged, 1),
                "revenue_eur":            round(revenue, 0),
                "fuel_cost_eur":          round(fuel_cost_eur, 0),
                "port_charges_eur":       round(port_charges_total, 0),
                "total_voyage_cost_eur":  round(total_cost, 0),
                "voyage_margin_eur":      round(voyage_margin, 0),
                "weather_delays_hrs":     weather_delay,
                "mechanical_delays_hrs":  mech_delay,
                "collections_count":      collections_count,
                "avg_collection_time_hrs": round(avg_collect_time, 1),
            })

            # Next voyage
            cur_time = arr_time + timedelta(hours=float(rng.uniform(6, 48)))
            cur_port = arr_port

    df = pd.DataFrame(records)
    df["departure_time"] = pd.to_datetime(df["departure_time"])
    df["arrival_time"]   = pd.to_datetime(df["arrival_time"])
    return df


# ============================================================
# Function 4 – Offshore Platform Data
# ============================================================

def generate_offshore_platform_data() -> pd.DataFrame:
    """
    ~5,000 rows: 8 platforms × ~625 days (2024-01-01 to 2026-06-01).
    """
    rng = np.random.default_rng(4242)

    platforms_df = _load_platforms()
    start = date(2024, 1, 1)
    end   = date(2026, 6, 1)
    all_dates = _date_range(start, end)

    records = []

    for _, plat in platforms_df.iterrows():
        pid         = plat["platform_id"]
        pname       = plat["platform_name"]
        region      = plat["region"]
        lat         = plat["latitude"]
        lon         = plat["longitude"]
        operator    = plat["operator"]
        base_prod   = plat["production_bpd"]
        base_wc     = plat["water_cut_pct"]
        storage_cap = plat["storage_capacity_m3"]
        coll_freq   = plat["collection_frequency_days"]
        access      = plat["access_restrictions"]
        min_dwt     = plat["minimum_tanker_size_dwt"]
        contract    = plat["contract_type"]
        dist_port   = plat["distance_from_nearest_port_nm"]

        # Classify field age: old = Brent Charlie / Forties / Ekofisk / Prinos
        is_old_field = any(x in pname for x in ["Brent", "Forties", "Ekofisk", "Prinos"])

        # Production decline & water cut increase per year
        prod_decline_rate = 0.02 if is_old_field else 0.005
        wc_increase_rate  = 0.02 if is_old_field else 0.05

        is_north_sea = "North Sea" in region

        # Storage state
        ref_year = 2024
        current_storage = rng.uniform(storage_cap * 0.2, storage_cap * 0.5)
        days_since_collection = 0
        last_collect_date = start - timedelta(days=int(rng.integers(1, coll_freq)))
        coll_interval = int(coll_freq + rng.integers(-5, 6))

        for d in all_dates:
            years_since_ref = (d - date(ref_year, 1, 1)).days / 365.25

            # Production and water cut trend
            prod_today = base_prod * (1 - prod_decline_rate) ** years_since_ref
            prod_today = max(prod_today * rng.uniform(0.97, 1.03), 100.0)

            wc_today = base_wc + wc_increase_rate * years_since_ref * 100
            wc_today = float(np.clip(wc_today, 0, 95))

            # Waste generation: production × water_cut → barrels water → m³ × waste_factor
            waste_factor = rng.uniform(0.1, 0.3)
            waste_gen = prod_today * (wc_today / 100) * 0.159 * waste_factor
            waste_gen = max(waste_gen, 0.5)

            # Storage accumulation
            current_storage += waste_gen
            days_since_collection += 1

            if days_since_collection >= coll_interval or current_storage >= storage_cap * 0.90:
                last_collect_date = d
                current_storage = storage_cap * rng.uniform(0.10, 0.25)
                days_since_collection = 0
                coll_interval = int(coll_freq + rng.integers(-5, 6))

            current_storage = min(current_storage, storage_cap)

            days_until_full = (storage_cap - current_storage) / max(waste_gen, 0.1)

            # Access restrictions: North Sea Nov-Mar = weather window only
            if is_north_sea and d.month in (11, 12, 1, 2, 3):
                acc_restriction = "Weather Window Only"
            else:
                acc_restriction = access

            records.append({
                "platform_id":                    pid,
                "platform_name":                  pname,
                "region":                         region,
                "latitude":                       lat,
                "longitude":                      lon,
                "operator":                       operator,
                "date":                           d.isoformat(),
                "production_bpd":                 round(prod_today, 0),
                "water_cut_pct":                  round(wc_today, 1),
                "waste_generation_m3_day":        round(waste_gen, 2),
                "storage_capacity_m3":            storage_cap,
                "current_storage_m3":             round(current_storage, 1),
                "days_until_full":                round(days_until_full, 1),
                "last_collection_date":           last_collect_date.isoformat(),
                "collection_frequency_days":      coll_freq,
                "access_restrictions":            acc_restriction,
                "minimum_tanker_size_dwt":        min_dwt,
                "contract_type":                  contract,
                "distance_from_nearest_port_nm":  dist_port,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ============================================================
# Function 5 – Maritime Weather
# ============================================================

def generate_maritime_weather() -> pd.DataFrame:
    """
    ~24,000 rows: 6 zones × 6-hourly × ~4.5 years.
    """
    rng = np.random.default_rng(7777)

    zones = ["MED-W", "MED-C", "MED-E", "ATL-GIB", "NORTH-SEA", "CHANNEL"]

    # Zone configs: (med_hs_summer, med_hs_winter, sst_mean, sst_amp)
    zone_config = {
        "MED-W":     {"hs_summer": (0.3, 1.5), "hs_winter": (1.0, 4.0), "sst_mean": 19, "sst_amp": 6, "is_north": False},
        "MED-C":     {"hs_summer": (0.3, 1.5), "hs_winter": (1.0, 4.0), "sst_mean": 20, "sst_amp": 6, "is_north": False},
        "MED-E":     {"hs_summer": (0.3, 1.5), "hs_winter": (1.0, 4.0), "sst_mean": 22, "sst_amp": 7, "is_north": False},
        "ATL-GIB":   {"hs_summer": (0.5, 2.0), "hs_winter": (1.5, 4.5), "sst_mean": 18, "sst_amp": 5, "is_north": False},
        "NORTH-SEA": {"hs_summer": (0.5, 2.5), "hs_winter": (2.0, 6.0), "sst_mean": 11, "sst_amp": 5, "is_north": True},
        "CHANNEL":   {"hs_summer": (0.4, 2.0), "hs_winter": (1.5, 5.0), "sst_mean": 13, "sst_amp": 5, "is_north": True},
    }

    start_ts = datetime(2022, 1, 1, 0, 0)
    end_ts   = datetime(2026, 6, 1, 0, 0)
    n_6hr    = int((end_ts - start_ts).total_seconds() / (6 * 3600))

    records = []

    for zone in zones:
        cfg = zone_config[zone]
        prev_hs = 1.0

        for i in range(n_6hr):
            ts = start_ts + timedelta(hours=i * 6)
            month = ts.month

            is_winter = month in (11, 12, 1, 2, 3)
            if is_winter:
                hs_lo, hs_hi = cfg["hs_winter"]
            else:
                hs_lo, hs_hi = cfg["hs_summer"]

            # Mean-reverting Hs
            hs_mid  = (hs_lo + hs_hi) / 2
            hs_noise = float(rng.normal(0, (hs_hi - hs_lo) * 0.15))
            hs = prev_hs + 0.1 * (hs_mid - prev_hs) + hs_noise
            hs = float(np.clip(hs, hs_lo * 0.6, hs_hi * 1.2))
            prev_hs = hs

            max_wave = hs * 1.8
            wave_period = 2.5 * math.sqrt(max(hs, 0.1))

            # Swell ~60-80% of Hs
            swell = hs * rng.uniform(0.55, 0.80)

            # Wind speed correlated with Beaufort
            beaufort = _hs_to_beaufort(hs)
            wind_mid = _beaufort_to_wind_knots(beaufort)
            wind_speed = float(np.clip(wind_mid * rng.uniform(0.88, 1.12), 0.5, 70.0))

            wind_dir = float(rng.uniform(0, 360))
            if cfg["is_north"]:
                # North Sea: prevailing SW-W
                wind_dir = float((220 + rng.normal(0, 50)) % 360)
            else:
                # Med: prevailing NW-W in summer
                wind_dir = float((280 + rng.normal(0, 70)) % 360)

            visibility = float(rng.uniform(0.5, 20.0))
            if is_winter and cfg["is_north"]:
                visibility = min(visibility, 10.0)

            # Sea surface temperature: sinusoidal
            sst = cfg["sst_mean"] + cfg["sst_amp"] * math.sin(
                math.pi * (month - 1) / 6 - math.pi / 2
            )
            sst += float(rng.normal(0, 0.5))
            sst = round(sst, 1)

            # Operation feasibility
            if hs < 2.0:
                feasibility = "Go"
            elif hs < 3.0:
                feasibility = "Marginal"
            else:
                feasibility = "No-Go"

            # Small vessel limit DWT
            if hs < 1.5:
                sv_limit = 0
            elif hs < 2.0:
                sv_limit = 500
            elif hs < 2.5:
                sv_limit = 1000
            elif hs < 3.0:
                sv_limit = 2000
            else:
                sv_limit = 99999  # all stop

            records.append({
                "timestamp":               ts.isoformat(),
                "zone_id":                 zone,
                "significant_wave_height_m": round(hs, 2),
                "max_wave_height_m":       round(max_wave, 2),
                "wave_period_s":           round(wave_period, 1),
                "swell_height_m":          round(swell, 2),
                "wind_speed_knots":        round(wind_speed, 1),
                "wind_direction_deg":      round(wind_dir, 0),
                "visibility_nm":           round(visibility, 1),
                "sea_surface_temp_c":      sst,
                "beaufort_scale":          beaufort,
                "operation_feasibility":   feasibility,
                "small_vessel_limit_dwt":  sv_limit,
            })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _hs_to_beaufort(hs: float) -> int:
    """Approximate Beaufort scale from significant wave height."""
    if hs < 0.1:   return 0
    if hs < 0.3:   return 1
    if hs < 0.6:   return 2
    if hs < 1.0:   return 3
    if hs < 1.5:   return 4
    if hs < 2.5:   return 5
    if hs < 3.5:   return 6
    if hs < 5.0:   return 7
    if hs < 7.0:   return 8
    return 9


def _beaufort_to_wind_knots(b: int) -> float:
    """Midpoint wind speed (knots) for each Beaufort number."""
    lookup = {0: 0, 1: 2, 2: 6, 3: 10, 4: 14, 5: 19, 6: 24, 7: 30, 8: 37, 9: 46}
    return float(lookup.get(b, 50))


# ============================================================
# Function 6 – Oil Market Data
# ============================================================

def generate_oil_market_data() -> pd.DataFrame:
    """
    ~1,600 rows: business days 2022-01-01 → 2026-06-01.
    """
    rng = np.random.default_rng(2022)

    start = date(2022, 1, 1)
    end   = date(2026, 6, 1)
    biz_days = pd.bdate_range(start=start, end=end)

    # Brent mean-reverting random walk
    brent_mean = 80.0
    mr_speed   = 0.02
    sigma_brent = 1.5

    # Simulate Ukraine-war spike: early 2022 → peak ~$115-120 in Q1 2022, gradual reversion
    brent = [85.0]  # starting price Jan 2022
    for i in range(1, len(biz_days)):
        prev  = brent[-1]
        shock = float(rng.normal(0, sigma_brent))
        # Target: spike to ~$120 by Mar 2022, then revert to $75-90
        d = biz_days[i].date()
        if d < date(2022, 4, 1):
            local_mean = 115.0  # war premium
        elif d < date(2022, 12, 1):
            local_mean = 90.0
        elif d < date(2023, 7, 1):
            local_mean = 78.0
        else:
            local_mean = brent_mean
        new_price = prev + mr_speed * (local_mean - prev) + shock
        new_price = float(np.clip(new_price, 55.0, 125.0))
        brent.append(new_price)

    # EUR/USD random walk
    eur_usd = [1.05]
    for _ in range(1, len(biz_days)):
        shock = float(rng.normal(0, 0.003))
        new_rate = eur_usd[-1] + 0.01 * (1.08 - eur_usd[-1]) + shock
        new_rate = float(np.clip(new_rate, 0.95, 1.20))
        eur_usd.append(new_rate)

    # Carbon credit: trend from €40 to €90
    n = len(biz_days)
    carbon_trend = np.linspace(40.0, 90.0, n)
    carbon_noise = rng.normal(0, 5.0, n)
    carbon = np.clip(carbon_trend + carbon_noise, 20.0, 110.0)

    # Shipping demand index
    sdi_base = 100.0
    sdi_trend = np.linspace(0, sdi_base * 0.02 * 4.5, n)  # +2%/year
    sdi = []
    for i, d in enumerate(biz_days):
        m = d.month
        # Peak Sep-Nov (month 9-11), low Feb-Apr
        seasonal = 5.0 * math.sin(math.pi * (m - 2) / 9)
        noise = float(rng.normal(0, 3))
        sdi.append(max(50.0, sdi_base + sdi_trend[i] + seasonal + noise))

    records = []
    for i, d in enumerate(biz_days):
        b = brent[i]
        hfo_380  = float(np.clip(b * 5.5 + rng.normal(0, 30), 280, 650))
        vlsfo    = float(np.clip(b * 7.5 + rng.normal(0, 40), 400, 850))
        mgo      = float(np.clip(b * 9.0 + rng.normal(0, 50), 600, 1200))

        rec_premium = float(rng.uniform(-15, 5))  # usually discount, mean ~-8%

        records.append({
            "date":                              d.date().isoformat(),
            "brent_crude_usd_bbl":               round(b, 2),
            "rotterdam_hfo_380_usd_mt":          round(hfo_380, 2),
            "rotterdam_vlsfo_usd_mt":            round(vlsfo, 2),
            "rotterdam_mgo_usd_mt":              round(mgo, 2),
            "recovered_fuel_premium_discount_pct": round(rec_premium, 2),
            "carbon_credit_eur_ton":             round(float(carbon[i]), 2),
            "eur_usd_rate":                      round(eur_usd[i], 4),
            "shipping_demand_index":             round(sdi[i], 2),
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ============================================================
# Function 7 – Orchestrator
# ============================================================

def generate_all() -> dict:
    """
    Generate and return all six datasets as a dict of DataFrames.
    Keys: port_waste_demand, fleet_status, voyage_history,
          offshore_platforms, maritime_weather, oil_market
    """
    print("Generating port waste demand...")
    demand = generate_port_waste_demand()

    print("Generating fleet status snapshots...")
    fleet = generate_hec_fleet_status()

    print("Generating voyage history...")
    voyages = generate_voyage_history(fleet_status_df=fleet, demand_df=demand)

    print("Generating offshore platform data...")
    platforms = generate_offshore_platform_data()

    print("Generating maritime weather data...")
    weather = generate_maritime_weather()

    print("Generating oil market data...")
    market = generate_oil_market_data()

    return {
        "port_waste_demand": demand,
        "fleet_status":      fleet,
        "voyage_history":    voyages,
        "offshore_platforms": platforms,
        "maritime_weather":  weather,
        "oil_market":        market,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = generate_all()
    for name, df in result.items():
        print(f"{name}: {len(df)} rows, {len(df.columns)} cols")
