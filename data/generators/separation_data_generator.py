"""
separation_data_generator.py

Generates all Use Case 1 source tables for the HEC AI Platform:
  - Waste Reception Log
  - Laboratory Analysis
  - Processing Batch
  - Process Sensor Data
  - Weather Conditions

Usage:
    from data.generators.separation_data_generator import generate_all
    result = generate_all()
    for name, df in result.items():
        print(f"{name}: {len(df)} rows, {len(df.columns)} cols")
"""

from __future__ import annotations

import math
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from data.generators.common import (
    BATCH_PRIORITIES,
    BATCH_PRIORITY_WEIGHTS,
    CLIENT_VESSEL_NAMES,
    COLLECTION_METHODS,
    COLLECTION_METHOD_WEIGHTS,
    CONTRACT_TYPES,
    CONTRACT_TYPE_WEIGHTS,
    DATE_END,
    DATE_START,
    FACILITIES,
    FUEL_TYPES,
    FUEL_TYPE_WEIGHTS,
    HEC_VESSEL_NAMES,
    MIXING_LEVELS,
    MIXING_LEVEL_WEIGHTS,
    PORTS,
    PRE_TREATMENTS,
    PRE_TREATMENT_WEIGHTS,
    RANDOM_SEED,
    SEPARATOR_TYPES,
    SEPARATOR_TYPE_WEIGHTS,
    TARGET_SPECS,
    TARGET_SPEC_WEIGHTS,
    VESSEL_TYPES,
    WASTE_CATEGORIES,
    generate_imo_number,
    seasonal_factor,
    temperature_for_location,
    walther_viscosity,
)

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_FACILITY_CODES = list(FACILITIES.keys())  # ["PIR", "HAM", "GIB", "MLT"]

# Waste subcategory distribution weights (sum to 1.0)
_WASTE_SUBCATEGORY_NAMES = [
    "Bilge Water",
    "Fuel Oil Sludge",
    "Oily Tank Washings",
    "Engine Room Oily Water",
    "Cargo Residues (Crude)",
    "Cargo Residues (Product)",
    "Offshore Production Slops",
    "Scale & Sludge from Tank Cleaning",
    "Dirty Ballast Water",
    "Vegetable Oil Residues",
    "Chemical Washing Residues",
]

_WASTE_SUBCATEGORY_WEIGHTS = [
    0.25,   # Bilge Water
    0.20,   # Fuel Oil Sludge
    0.15,   # Oily Tank Washings
    0.12,   # Engine Room Oily Water
    0.08,   # Cargo Residues (Crude)
    0.06,   # Cargo Residues (Product)
    0.05,   # Offshore Production Slops
    0.04,   # Scale & Sludge from Tank Cleaning
    0.03,   # Dirty Ballast Water
    0.015,  # Vegetable Oil Residues
    0.005,  # Chemical Washing Residues
]

# MARPOL Annex classification per subcategory
_MARPOL_ANNEX = {
    "Bilge Water": "Annex I - Oily Residues",
    "Fuel Oil Sludge": "Annex I - Oily Residues",
    "Oily Tank Washings": "Annex I - Oily Residues",
    "Engine Room Oily Water": "Annex I - Oily Residues",
    "Cargo Residues (Crude)": "Annex I - Oily Residues",
    "Cargo Residues (Product)": "Annex I - Oily Residues",
    "Offshore Production Slops": "Annex I - Oily Residues",
    "Scale & Sludge from Tank Cleaning": "Annex I - Oily Residues",
    "Dirty Ballast Water": "Annex I - Oily Residues",
    "Vegetable Oil Residues": "Annex II - Noxious Liquid",
    "Chemical Washing Residues": "Annex II - Noxious Liquid",
}

# Volume ranges per subcategory (m³)
_VOLUME_RANGES = {
    "Bilge Water":                       (10,   80),
    "Fuel Oil Sludge":                   (20,  200),
    "Oily Tank Washings":                (50,  500),
    "Engine Room Oily Water":            (10,   80),
    "Cargo Residues (Crude)":           (100, 2500),
    "Cargo Residues (Product)":         (100, 2500),
    "Offshore Production Slops":        (200, 2000),
    "Scale & Sludge from Tank Cleaning": (10,  150),
    "Dirty Ballast Water":               (20,  300),
    "Vegetable Oil Residues":            (20,  200),
    "Chemical Washing Residues":         (10,   80),
}

# Acceptance fee per m³ (€) — more difficult/viscous waste is pricier
_ACCEPTANCE_FEE_RANGE = {
    "Bilge Water":                       (15,  25),
    "Fuel Oil Sludge":                   (30,  50),
    "Oily Tank Washings":                (25,  40),
    "Engine Room Oily Water":            (18,  28),
    "Cargo Residues (Crude)":           (22,  38),
    "Cargo Residues (Product)":         (20,  35),
    "Offshore Production Slops":        (35,  50),
    "Scale & Sludge from Tank Cleaning": (40,  50),
    "Dirty Ballast Water":               (15,  22),
    "Vegetable Oil Residues":            (25,  40),
    "Chemical Washing Residues":         (40,  50),
}

# Vessel type → preferred waste subcategories with sampling weights
_VESSEL_WASTE_AFFINITY: Dict[str, list] = {
    "Crude Oil Tanker":     ["Cargo Residues (Crude)", "Oily Tank Washings", "Fuel Oil Sludge"],
    "Product Tanker":       ["Cargo Residues (Product)", "Oily Tank Washings", "Fuel Oil Sludge"],
    "Chemical Tanker":      ["Chemical Washing Residues", "Cargo Residues (Product)", "Oily Tank Washings"],
    "LNG Carrier":          ["Fuel Oil Sludge", "Bilge Water", "Engine Room Oily Water"],
    "Bulk Carrier":         ["Fuel Oil Sludge", "Bilge Water", "Engine Room Oily Water"],
    "Container Ship":       ["Fuel Oil Sludge", "Bilge Water", "Engine Room Oily Water"],
    "General Cargo":        ["Bilge Water", "Engine Room Oily Water", "Fuel Oil Sludge"],
    "Ro-Ro / Ferry":        ["Bilge Water", "Engine Room Oily Water", "Fuel Oil Sludge"],
    "Offshore Supply Vessel": ["Offshore Production Slops", "Fuel Oil Sludge", "Bilge Water"],
    "Fishing Vessel":       ["Bilge Water", "Engine Room Oily Water", "Fuel Oil Sludge"],
}

# Facility → ports in service area
_FACILITY_PORTS = {
    "PIR": ["PIR", "GEN", "MRS", "LIM", "ALE", "IST", "PSD"],
    "HAM": ["HAM", "RTM"],
    "GIB": ["GIB", "ALG", "TAN", "BCN"],
    "MLT": ["MLT", "GEN", "MRS", "LIM"],
}

# Facility volume distribution weights (PIR 45%, HAM 24%, GIB 18%, MLT 13%)
_FACILITY_VOLUME_WEIGHTS = [0.45, 0.24, 0.18, 0.13]

# Base yield ranges per subcategory (min, max) in percent
_BASE_YIELD_RANGES = {
    "Cargo Residues (Product)":          (85, 95),
    "Cargo Residues (Crude)":            (75, 90),
    "Offshore Production Slops":         (55, 75),
    "Oily Tank Washings":                (50, 70),
    "Fuel Oil Sludge":                   (40, 65),
    "Bilge Water":                       (30, 55),
    "Engine Room Oily Water":            (25, 50),
    "Vegetable Oil Residues":            (60, 80),
    "Chemical Washing Residues":         (20, 45),
    "Scale & Sludge from Tank Cleaning": (20, 45),
    "Dirty Ballast Water":               (20, 45),
}

# Pre-treatment yield bonus
_PRE_TREATMENT_BONUS = {
    "None":                    0.0,
    "Heating":                 3.0,
    "Gravity Separation":      4.0,
    "Centrifugation":          3.5,
    "Chemical Demulsification": 5.0,
    "Filtration":              2.0,
}

# Operator pools per facility: (facility, count, experience range years)
_OPERATOR_CONFIG = {
    "PIR": (10, (1, 25)),
    "HAM": (8,  (1, 25)),
    "GIB": (7,  (1, 25)),
    "MLT": (5,  (1, 25)),
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _make_rng(seed=RANDOM_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_datetime(rng: np.random.Generator, start: datetime, end: datetime) -> datetime:
    """Return a random datetime between start and end."""
    delta_s = int((end - start).total_seconds())
    offset_s = int(rng.integers(0, delta_s))
    return start + timedelta(seconds=offset_s)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _weighted_choice(rng: np.random.Generator, items: list, weights: list):
    """Single weighted choice using numpy."""
    idx = rng.choice(len(items), p=np.array(weights) / np.sum(weights))
    return items[idx]


def _weighted_choices(rng: np.random.Generator, items: list, weights: list, n: int) -> np.ndarray:
    """n weighted choices using numpy."""
    p = np.array(weights, dtype=float)
    p /= p.sum()
    return rng.choice(len(items), size=n, p=p)


# ---------------------------------------------------------------------------
# Function 1: generate_waste_reception_log
# ---------------------------------------------------------------------------

def generate_waste_reception_log(n: int = 15000) -> pd.DataFrame:
    """
    Generate waste reception log — one row per waste collection event.
    """
    rng = _make_rng(RANDOM_SEED)

    dt_start = datetime(DATE_START.year, DATE_START.month, DATE_START.day)
    dt_end   = datetime(DATE_END.year,   DATE_END.month,   DATE_END.day)

    # Pre-build operator pools per facility
    operators = {}
    for fac, (count, exp_range) in _OPERATOR_CONFIG.items():
        operators[fac] = [f"OP-{fac}-{i+1:02d}" for i in range(count)]

    # Build facility rows proportional to volume weights with seasonal adjustment
    # First pass: assign facility codes by volume weight
    facility_indices = _weighted_choices(
        rng, _FACILITY_CODES, _FACILITY_VOLUME_WEIGHTS, n
    )

    # Generate timestamps spread across the date range
    total_seconds = int((dt_end - dt_start).total_seconds())
    raw_offsets = rng.integers(0, total_seconds, size=n)
    timestamps = [dt_start + timedelta(seconds=int(s)) for s in raw_offsets]
    timestamps.sort()  # chronological order

    # Re-assign facility based on seasonal factor biasing
    # After sorting timestamps, reassign facility with seasonal weighting
    facility_codes = []
    for ts in timestamps:
        month_factors = []
        for fac in _FACILITY_CODES:
            base = _FACILITY_VOLUME_WEIGHTS[_FACILITY_CODES.index(fac)]
            sf   = seasonal_factor(ts.date(), fac)
            month_factors.append(base * sf)
        total = sum(month_factors)
        p = [x / total for x in month_factors]
        idx = int(rng.choice(len(_FACILITY_CODES), p=p))
        facility_codes.append(_FACILITY_CODES[idx])

    # Vessel types
    vessel_type_list = list(VESSEL_TYPES.keys())
    vessel_type_weights = [1 / len(vessel_type_list)] * len(vessel_type_list)
    vessel_type_weights[0] = 1.8   # Crude Oil Tanker more common
    vessel_type_weights[4] = 1.5   # Bulk Carrier more common
    vessel_type_weights[5] = 1.3   # Container Ship
    vessel_type_weights[6] = 1.2   # General Cargo

    vt_indices = _weighted_choices(rng, vessel_type_list, vessel_type_weights, n)
    source_vessel_types = [vessel_type_list[i] for i in vt_indices]

    # Fuel types
    fuel_type_indices = _weighted_choices(rng, FUEL_TYPES, FUEL_TYPE_WEIGHTS, n)
    source_fuel_types = [FUEL_TYPES[i] for i in fuel_type_indices]

    # Collecting vessel names
    collecting_vessel_names = [
        HEC_VESSEL_NAMES[int(rng.integers(0, len(HEC_VESSEL_NAMES)))]
        for _ in range(n)
    ]

    # Source vessel names
    source_vessel_names_pool = CLIENT_VESSEL_NAMES + [
        f"MV {rng.integers(1000,9999)}" for _ in range(50)
    ]
    source_vessel_names = [
        source_vessel_names_pool[int(rng.integers(0, len(source_vessel_names_pool)))]
        for _ in range(n)
    ]

    # IMO numbers (unique-ish, one per vessel)
    source_vessel_imos = [generate_imo_number(None) for _ in range(n)]

    # Waste subcategories with vessel-type affinity
    waste_subcats = []
    for vtype in source_vessel_types:
        affinities = _VESSEL_WASTE_AFFINITY.get(vtype, _WASTE_SUBCATEGORY_NAMES)
        # 60% chance: pick from affinity list; 40%: random from all
        if rng.random() < 0.60:
            affinity_weights = [1.0] * len(affinities)
            subcat = _weighted_choice(rng, affinities, affinity_weights)
        else:
            idx = int(_weighted_choices(rng, _WASTE_SUBCATEGORY_NAMES, _WASTE_SUBCATEGORY_WEIGHTS, 1)[0])
            subcat = _WASTE_SUBCATEGORY_NAMES[idx]
        waste_subcats.append(subcat)

    # Vessel specs
    source_vessel_gts = []
    source_vessel_dwts = []
    source_vessel_engine_kws = []
    for vtype in source_vessel_types:
        specs = VESSEL_TYPES[vtype]
        gt_lo, gt_hi = specs["gt_range"]
        ek_lo, ek_hi = specs["engine_kw_range"]
        gt = float(rng.integers(gt_lo, gt_hi + 1))
        # DWT roughly 1.5x GT for tankers, 0.8x for others
        dwt_factor = 1.5 if "Tanker" in vtype else 0.8
        dwt = round(gt * dwt_factor * float(rng.uniform(0.85, 1.15)))
        ek = float(rng.integers(ek_lo, ek_hi + 1))
        source_vessel_gts.append(gt)
        source_vessel_dwts.append(dwt)
        source_vessel_engine_kws.append(ek)

    # Declared volumes
    declared_volumes = []
    for subcat in waste_subcats:
        lo, hi = _VOLUME_RANGES[subcat]
        # Vessel type waste_factor multiplier
        vol = float(rng.uniform(lo, hi))
        declared_volumes.append(round(vol, 1))

    # Actual volumes: ±10% of declared
    actual_volumes = [
        round(float(dv * rng.uniform(0.90, 1.10)), 1)
        for dv in declared_volumes
    ]

    # Waste temperatures — ambient at facility ± variability
    waste_temperatures = []
    for i, ts in enumerate(timestamps):
        fac = facility_codes[i]
        amb = temperature_for_location(ts.date(), fac)
        # Waste arrives warmer due to engine heat; sludges can be heated
        subcat = waste_subcats[i]
        if subcat in ("Fuel Oil Sludge", "Oily Tank Washings", "Cargo Residues (Crude)"):
            temp = amb + float(rng.uniform(15, 35))
        elif subcat in ("Cargo Residues (Product)", "Offshore Production Slops"):
            temp = amb + float(rng.uniform(10, 25))
        else:
            temp = amb + float(rng.uniform(2, 10))
        waste_temperatures.append(round(temp, 1))

    # Source ports
    source_ports = []
    for fac in facility_codes:
        port_options = _FACILITY_PORTS[fac]
        port = port_options[int(rng.integers(0, len(port_options)))]
        source_ports.append(port)

    # Collection methods
    cm_indices = _weighted_choices(rng, COLLECTION_METHODS, COLLECTION_METHOD_WEIGHTS, n)
    collection_methods = [COLLECTION_METHODS[i] for i in cm_indices]

    # Mixing during transport
    mix_indices = _weighted_choices(rng, MIXING_LEVELS, MIXING_LEVEL_WEIGHTS, n)
    mixing_levels = [MIXING_LEVELS[i] for i in mix_indices]

    # Days in tank before collection
    days_in_tank = [int(rng.integers(0, 30)) for _ in range(n)]

    # Customer IDs (100 customers)
    customer_ids = [f"CUST-{int(rng.integers(1000, 9999))}" for _ in range(n)]

    # Contract types
    ct_indices = _weighted_choices(rng, CONTRACT_TYPES, CONTRACT_TYPE_WEIGHTS, n)
    contract_types = [CONTRACT_TYPES[i] for i in ct_indices]

    # Acceptance fees
    acceptance_fees = []
    for subcat, vol in zip(waste_subcats, actual_volumes):
        lo, hi = _ACCEPTANCE_FEE_RANGE[subcat]
        fee_per_m3 = float(rng.uniform(lo, hi))
        acceptance_fees.append(round(fee_per_m3 * vol, 2))

    # MARPOL annex classification
    marpol_annexes = [_MARPOL_ANNEX[s] for s in waste_subcats]

    # Collecting vessel DWT (HEC vessels are smaller, 500-5000 DWT)
    collecting_vessel_dwts = [
        int(rng.integers(500, 5001)) for _ in range(n)
    ]

    # Build sequence numbers per facility per year
    seq_counters: Dict[str, int] = {}
    reception_ids = []
    for i, ts in enumerate(timestamps):
        fac = facility_codes[i]
        year = ts.year
        key = f"{fac}-{year}"
        seq_counters[key] = seq_counters.get(key, 0) + 1
        reception_ids.append(f"REC-{fac}-{year}-{seq_counters[key]:05d}")

    df = pd.DataFrame({
        "reception_id":                reception_ids,
        "timestamp":                   timestamps,
        "facility_code":               facility_codes,
        "collecting_vessel_name":      collecting_vessel_names,
        "collecting_vessel_dwt":       collecting_vessel_dwts,
        "source_vessel_name":          source_vessel_names,
        "source_vessel_imo":           source_vessel_imos,
        "source_vessel_type":          source_vessel_types,
        "source_vessel_gt":            source_vessel_gts,
        "source_vessel_engine_kw":     source_vessel_engine_kws,
        "source_vessel_fuel_type":     source_fuel_types,
        "source_port":                 source_ports,
        "waste_category_marpol":       marpol_annexes,
        "waste_subcategory":           waste_subcats,
        "declared_volume_m3":          declared_volumes,
        "actual_volume_m3":            actual_volumes,
        "waste_temperature_c":         waste_temperatures,
        "collection_method":           collection_methods,
        "mixing_during_transport":     mixing_levels,
        "days_in_tank_before_collection": days_in_tank,
        "customer_id":                 customer_ids,
        "contract_type":               contract_types,
        "acceptance_fee_eur":          acceptance_fees,
        # add DWT for source vessel (rename for clarity)
        "source_vessel_dwt":           source_vessel_dwts,
    })

    # Reorder to match spec exactly
    df = df[[
        "reception_id", "timestamp", "facility_code",
        "collecting_vessel_name", "collecting_vessel_dwt",
        "source_vessel_name", "source_vessel_imo", "source_vessel_type",
        "source_vessel_gt", "source_vessel_engine_kw", "source_vessel_fuel_type",
        "source_port", "waste_category_marpol", "waste_subcategory",
        "declared_volume_m3", "actual_volume_m3", "waste_temperature_c",
        "collection_method", "mixing_during_transport",
        "days_in_tank_before_collection", "customer_id", "contract_type",
        "acceptance_fee_eur",
    ]]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Function 2: generate_laboratory_analysis
# ---------------------------------------------------------------------------

def generate_laboratory_analysis(reception_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate lab analysis data: 3 samples per reception (Inlet, Mid-Process, Oil Output).
    """
    rng = _make_rng(RANDOM_SEED + 1)

    rows = []
    sample_seq: Dict[str, int] = {}

    # Pre-build a lookup for WASTE_CATEGORIES
    cat_keys = list(WASTE_CATEGORIES.keys())

    for _, rec in reception_df.iterrows():
        rec_id     = rec["reception_id"]
        facility   = rec["facility_code"]
        ts         = rec["timestamp"]
        subcat     = rec["waste_subcategory"]
        fuel_type  = rec["source_vessel_fuel_type"]
        year       = ts.year if isinstance(ts, datetime) else pd.Timestamp(ts).year

        key = f"{facility}-{year}"
        seq_sample = sample_seq.get(key, 0)

        # Resolve WASTE_CATEGORIES entry (handles minor name mismatches)
        cat_info = WASTE_CATEGORIES.get(subcat)
        if cat_info is None:
            # fallback to Bilge Water
            cat_info = WASTE_CATEGORIES["Bilge Water"]

        oil_lo,  oil_hi  = cat_info["oil_pct_range"]
        wat_lo,  wat_hi  = cat_info["water_pct_range"]
        sol_lo,  sol_hi  = cat_info["solids_pct_range"]

        # INLET sample  — draw from category ranges and normalise to 100%
        inlet_oil_raw  = float(rng.uniform(oil_lo, oil_hi))
        inlet_wat_raw  = float(rng.uniform(wat_lo, wat_hi))
        inlet_sol_raw  = float(rng.uniform(sol_lo, sol_hi))
        raw_sum = inlet_oil_raw + inlet_wat_raw + inlet_sol_raw
        inlet_oil  = round(inlet_oil_raw  / raw_sum * 100 + rng.uniform(-0.5, 0.5), 2)
        inlet_wat  = round(inlet_wat_raw  / raw_sum * 100 + rng.uniform(-0.5, 0.5), 2)
        inlet_sol  = round(100 - inlet_oil - inlet_wat, 2)
        inlet_sol  = max(0.01, inlet_sol)

        # MID-PROCESS sample — partial separation
        mid_oil_boost = float(rng.uniform(5, 20))
        mid_oil  = _clamp(inlet_oil + mid_oil_boost, inlet_oil, 99.0)
        mid_wat  = _clamp(inlet_wat - mid_oil_boost * 0.8, 0.5, 99.0)
        mid_sol  = _clamp(100 - mid_oil - mid_wat, 0.01, 30.0)
        # re-normalise
        _s = mid_oil + mid_wat + mid_sol
        mid_oil = round(mid_oil / _s * 100, 2)
        mid_wat = round(mid_wat / _s * 100, 2)
        mid_sol = round(100 - mid_oil - mid_wat, 2)

        # OIL OUTPUT sample
        out_oil = round(float(rng.uniform(92, 99)), 2)
        out_wat = round(float(rng.uniform(0.1, 3.0)), 2)
        out_sol = round(100 - out_oil - out_wat, 2)
        out_sol = max(0.01, out_sol)

        # Viscosity profile based on subcategory
        if subcat in ("Bilge Water", "Engine Room Oily Water", "Dirty Ballast Water"):
            visc_40_lo, visc_40_hi = 2, 50
        elif subcat in ("Fuel Oil Sludge", "Scale & Sludge from Tank Cleaning"):
            visc_40_lo, visc_40_hi = 3000, 15000
        elif subcat in ("Cargo Residues (Crude)", "Offshore Production Slops",
                        "Oily Tank Washings"):
            visc_40_lo, visc_40_hi = 50, 500
        else:
            visc_40_lo, visc_40_hi = 10, 300

        inlet_visc40 = float(rng.uniform(visc_40_lo, visc_40_hi))
        inlet_visc100 = round(walther_viscosity(inlet_visc40, 100.0), 2)
        mid_visc40 = round(inlet_visc40 * float(rng.uniform(0.4, 0.7)), 2)
        mid_visc100 = round(walther_viscosity(mid_visc40, 100.0), 2)
        out_visc40 = round(float(rng.uniform(5, 80)), 2)
        out_visc100 = round(walther_viscosity(out_visc40, 100.0), 2)

        # Flash point
        if subcat in ("Cargo Residues (Product)", "Vegetable Oil Residues"):
            fp_lo, fp_hi = 21, 60
        elif subcat in ("Bilge Water", "Engine Room Oily Water", "Dirty Ballast Water"):
            fp_lo, fp_hi = 30, 80
        elif subcat in ("Fuel Oil Sludge", "Scale & Sludge from Tank Cleaning"):
            fp_lo, fp_hi = 80, 200
        else:
            fp_lo, fp_hi = 60, 150

        inlet_fp = round(float(rng.uniform(fp_lo, fp_hi)), 1)
        out_fp   = round(float(rng.uniform(fp_lo, fp_hi + 20)), 1)

        # Sulfur content based on fuel type
        if fuel_type == "HFO":
            sulfur_lo, sulfur_hi = 2.0, 4.5
        elif fuel_type == "VLSFO":
            sulfur_lo, sulfur_hi = 0.1, 0.5
        elif fuel_type in ("MGO", "MDO"):
            sulfur_lo, sulfur_hi = 0.05, 0.1
        else:
            sulfur_lo, sulfur_hi = 0.01, 0.3

        inlet_sulfur = round(float(rng.uniform(sulfur_lo, sulfur_hi)), 4)
        out_sulfur   = round(inlet_sulfur * float(rng.uniform(0.9, 1.05)), 4)

        # Vanadium based on fuel type
        if fuel_type == "HFO":
            van_lo, van_hi = 100, 600
        elif fuel_type == "VLSFO":
            van_lo, van_hi = 10, 50
        else:
            van_lo, van_hi = 1, 5

        inlet_vanadium = round(float(rng.uniform(van_lo, van_hi)), 1)

        # Nickel ~ Vanadium/3
        inlet_nickel = round(inlet_vanadium / 3 * float(rng.uniform(0.7, 1.3)), 1)

        # Iron 10-500 ppm in inlet
        inlet_iron = round(float(rng.uniform(10, 500)), 1)

        # Sodium: high in seawater-contaminated waste
        if subcat in ("Bilge Water", "Dirty Ballast Water", "Engine Room Oily Water"):
            sodium_lo, sodium_hi = 5000, 30000
        else:
            sodium_lo, sodium_hi = 50, 500

        inlet_sodium = round(float(rng.uniform(sodium_lo, sodium_hi)), 1)

        # Al+Si: contamination indicator
        inlet_alsi = round(float(rng.uniform(5, 80)), 1)

        # Chlorides
        if subcat in ("Bilge Water", "Dirty Ballast Water"):
            inlet_chloride = round(float(rng.uniform(5000, 25000)), 1)
        else:
            inlet_chloride = round(float(rng.uniform(50, 500)), 1)

        # Ash content
        inlet_ash = round(float(rng.uniform(0.05, 3.0)), 3)

        # Carbon residue
        inlet_carbon_residue = round(float(rng.uniform(2.0, 18.0)), 2)

        # Pour point
        inlet_pour_point = round(float(rng.uniform(-15, 30)), 1)

        # Acid number
        inlet_acid_number = round(float(rng.uniform(0.1, 5.0)), 3)

        # Calorific value: proportional to oil content (pure oil ~42 MJ/kg)
        def calorific_gross(oil_pct: float) -> float:
            return round(42.0 * oil_pct / 100 * float(rng.uniform(0.95, 1.05)), 2)

        inlet_cv_gross = calorific_gross(inlet_oil)
        inlet_cv_net   = round(inlet_cv_gross * float(rng.uniform(0.93, 0.96)), 2)
        mid_cv_gross   = calorific_gross(mid_oil)
        mid_cv_net     = round(mid_cv_gross * float(rng.uniform(0.93, 0.96)), 2)
        out_cv_gross   = calorific_gross(out_oil)
        out_cv_net     = round(out_cv_gross * float(rng.uniform(0.93, 0.96)), 2)

        # PCB (trace), BTEX, H2S
        pcb_inlet   = round(float(rng.uniform(0, 0.1)), 4)
        btex_inlet  = round(float(rng.uniform(0, 50)), 2)
        h2s_inlet   = round(float(rng.uniform(0, 200)), 2)

        # Emulsion layer
        if subcat in ("Fuel Oil Sludge", "Oily Tank Washings", "Offshore Production Slops"):
            emulsion_lo, emulsion_hi = 5, 60
        else:
            emulsion_lo, emulsion_hi = 0, 10

        inlet_emulsion = round(float(rng.uniform(emulsion_lo, emulsion_hi)), 1)

        # Particle size D50 micron
        if subcat in ("Scale & Sludge from Tank Cleaning", "Fuel Oil Sludge"):
            ps_lo, ps_hi = 20, 200
        else:
            ps_lo, ps_hi = 1, 50

        inlet_ps = round(float(rng.uniform(ps_lo, ps_hi)), 1)

        # Density
        def _density(oil_pct: float, water_pct: float) -> float:
            """Weighted density estimate kg/m³ at 15°C."""
            oil_d   = float(rng.uniform(820, 920))
            water_d = float(rng.uniform(1000, 1025))
            sol_d   = float(rng.uniform(1200, 1500))
            sol_pct = max(0, 100 - oil_pct - water_pct)
            d = (oil_pct * oil_d + water_pct * water_d + sol_pct * sol_d) / 100
            return round(d, 1)

        inlet_density  = _density(inlet_oil, inlet_wat)
        mid_density    = _density(mid_oil, mid_wat)
        out_density    = _density(out_oil, out_wat)

        # Build 3 sample points, optionally 2 more (Water Output, Solids Output)
        sample_points = ["Inlet", "Mid-Process", "Oil Output"]
        sample_data = [
            # (oil, water, sol, visc40, visc100, fp, density, cv_gross, cv_net)
            (inlet_oil, inlet_wat, inlet_sol, inlet_visc40,  inlet_visc100,
             inlet_fp, inlet_density, inlet_cv_gross, inlet_cv_net),
            (mid_oil,   mid_wat,   mid_sol,  mid_visc40,   mid_visc100,
             round(inlet_fp * 0.95, 1), mid_density, mid_cv_gross, mid_cv_net),
            (out_oil,   out_wat,   out_sol,  out_visc40,   out_visc100,
             out_fp, out_density, out_cv_gross, out_cv_net),
        ]

        # Occasionally add Water Output and Solids Output samples (30% chance)
        if rng.random() < 0.30:
            # Water output: water-dominant
            wo_oil = round(float(rng.uniform(0.001, 0.05)), 4)
            wo_wat = round(100 - wo_oil - 0.01, 2)
            wo_sol = 0.01
            sample_points.append("Water Output")
            sample_data.append((
                wo_oil, wo_wat, wo_sol,
                round(float(rng.uniform(0.5, 5.0)), 2),
                round(float(rng.uniform(0.5, 2.0)), 2),
                None,
                round(float(rng.uniform(998, 1025)), 1),
                calorific_gross(wo_oil), calorific_gross(wo_oil) * 0.93,
            ))
        if rng.random() < 0.20:
            # Solids output
            so_oil = round(float(rng.uniform(1, 15)), 2)
            so_wat = round(float(rng.uniform(5, 25)), 2)
            so_sol = round(100 - so_oil - so_wat, 2)
            sample_points.append("Solids Output")
            sample_data.append((
                so_oil, so_wat, so_sol,
                None, None, None,
                round(float(rng.uniform(1200, 1800)), 1),
                calorific_gross(so_oil), calorific_gross(so_oil) * 0.93,
            ))

        for sp_idx, (sp, sd) in enumerate(zip(sample_points, sample_data)):
            seq_sample += 1
            sid = f"LAB-{facility}-{year}-{seq_sample:05d}"
            oil_c, wat_c, sol_c, v40, v100, fp_v, dens, cv_g, cv_n = sd

            # Sample timestamp: offset from reception timestamp
            samp_ts = pd.Timestamp(ts) + timedelta(
                hours=int(rng.integers(0, 24 * 2))
            )

            rows.append({
                "sample_id":                  sid,
                "reception_id":               rec_id,
                "sample_point":               sp,
                "sample_timestamp":           samp_ts,
                "water_content_pct":          round(float(wat_c), 2),
                "oil_content_pct":            round(float(oil_c), 2),
                "solids_content_pct":         round(float(sol_c), 2),
                "density_15c_kg_m3":          round(float(dens), 1),
                "viscosity_40c_cst":          round(float(v40), 2) if v40 is not None else None,
                "viscosity_100c_cst":         round(float(v100), 2) if v100 is not None else None,
                "flash_point_c":              round(float(fp_v), 1) if fp_v is not None else None,
                "pour_point_c":               inlet_pour_point,
                "sulfur_total_pct":           round(float(inlet_sulfur * (0.8 if sp == "Oil Output" else 1.0)), 4),
                "ash_content_pct":            round(float(inlet_ash * (0.6 if sp == "Oil Output" else 1.0)), 3),
                "carbon_residue_pct":         round(float(inlet_carbon_residue), 2),
                "vanadium_ppm":               round(float(inlet_vanadium * (0.9 if sp == "Oil Output" else 1.0)), 1),
                "nickel_ppm":                 round(float(inlet_nickel  * (0.9 if sp == "Oil Output" else 1.0)), 1),
                "iron_ppm":                   round(float(inlet_iron    * (0.5 if sp == "Oil Output" else 1.0)), 1),
                "sodium_ppm":                 round(float(inlet_sodium  * (0.1 if sp == "Oil Output" else 1.0)), 1),
                "aluminum_silicon_ppm":       round(float(inlet_alsi    * (0.3 if sp == "Oil Output" else 1.0)), 1),
                "chloride_ppm":               round(float(inlet_chloride * (0.05 if sp == "Oil Output" else 1.0)), 1),
                "acid_number_mg_koh_g":       round(float(inlet_acid_number), 3),
                "calorific_value_gross_mj_kg": round(float(cv_g), 2),
                "calorific_value_net_mj_kg":   round(float(cv_n), 2),
                "pcb_ppm":                    round(float(pcb_inlet), 4),
                "btex_ppm":                   round(float(btex_inlet), 2),
                "h2s_ppm":                    round(float(h2s_inlet * (0.1 if sp == "Oil Output" else 1.0)), 2),
                "emulsion_layer_pct":         round(float(inlet_emulsion * (0.1 if sp == "Oil Output" else 1.0)), 1),
                "particle_size_d50_micron":   round(float(inlet_ps), 1),
            })

        sample_seq[key] = seq_sample

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Function 3: generate_processing_batch
# ---------------------------------------------------------------------------

def generate_processing_batch(
    reception_df: pd.DataFrame,
    lab_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate ~12,000 processing batch rows. Most receptions map 1:1 to batches;
    some combine 2-3 similar receptions.
    """
    rng = _make_rng(RANDOM_SEED + 2)

    # Load equipment master
    equip_path = os.path.join(
        os.path.dirname(__file__), "..", "reference", "equipment_master.csv"
    )
    try:
        equip_df = pd.read_csv(equip_path)
    except FileNotFoundError:
        # Build minimal equipment table from FACILITIES
        equip_df = pd.DataFrame([
            {"unit_id": f"{fac}-DSC-001", "facility_code": fac,
             "equipment_type": "3-Phase Disc Stack Centrifuge",
             "condition_score": 8.0, "capacity_m3_hr": 25.0}
            for fac in _FACILITY_CODES
        ])

    # Build per-facility equipment lookup
    equip_by_fac: Dict[str, list] = {}
    for _, row in equip_df.iterrows():
        fac = row["facility_code"]
        equip_by_fac.setdefault(fac, []).append(row)

    # Build operator pools
    op_pools: Dict[str, list] = {}
    op_exp_map: Dict[str, float] = {}
    for fac, (count, (exp_lo, exp_hi)) in _OPERATOR_CONFIG.items():
        pool = []
        for i in range(count):
            op_id = f"OP-{fac}-{i+1:02d}"
            exp   = float(rng.uniform(exp_lo, exp_hi))
            pool.append(op_id)
            op_exp_map[op_id] = exp
        op_pools[fac] = pool

    # Prepare inlet lab data lookup: reception_id → (oil_content, emulsion, viscosity40)
    inlet_lab = (
        lab_df[lab_df["sample_point"] == "Inlet"]
        .set_index("reception_id")
        [["oil_content_pct", "emulsion_layer_pct", "viscosity_40c_cst"]]
    )

    # Group receptions by facility for potential batch-combining
    rec_by_fac: Dict[str, list] = {}
    for _, row in reception_df.iterrows():
        rec_by_fac.setdefault(row["facility_code"], []).append(row)

    rows = []
    batch_seq: Dict[str, int] = {}

    for fac in _FACILITY_CODES:
        recs = rec_by_fac.get(fac, [])
        if not recs:
            continue

        i = 0
        while i < len(recs):
            # Decide: combine 2-3 similar receptions (15% chance, up to 3)
            combine = 1
            if i + 1 < len(recs) and rng.random() < 0.15:
                combine = int(rng.integers(2, 4))
                combine = min(combine, len(recs) - i)

            batch_recs = recs[i: i + combine]
            i += combine

            # Primary reception
            primary = batch_recs[0]
            ts_start = pd.Timestamp(primary["timestamp"])

            # Total input volume
            total_vol = sum(r["actual_volume_m3"] for r in batch_recs)
            reception_ids_str = ";".join(r["reception_id"] for r in batch_recs)

            # Get inlet lab data for primary reception
            rec_id = primary["reception_id"]
            if rec_id in inlet_lab.index:
                lab_row = inlet_lab.loc[rec_id]
                oil_content   = float(lab_row["oil_content_pct"]) / 100
                emulsion_pct  = float(lab_row["emulsion_layer_pct"]) if pd.notna(lab_row["emulsion_layer_pct"]) else 5.0
                visc40        = float(lab_row["viscosity_40c_cst"])  if pd.notna(lab_row["viscosity_40c_cst"])  else 100.0
            else:
                oil_content  = 0.30
                emulsion_pct = 5.0
                visc40       = 100.0

            subcat   = primary["waste_subcategory"]
            feed_temp = float(primary["waste_temperature_c"])

            # Pick separator unit
            fac_equip = equip_by_fac.get(fac, [])
            if fac_equip:
                equip_row = fac_equip[int(rng.integers(0, len(fac_equip)))]
                sep_unit_id = equip_row["unit_id"]
                cond_score  = float(equip_row["condition_score"]) if pd.notna(equip_row.get("condition_score", 8.0)) else 8.0
            else:
                sep_unit_id = f"{fac}-SEP-001"
                cond_score  = 8.0

            # Pre-treatment
            pt_idx  = int(_weighted_choices(rng, PRE_TREATMENTS, PRE_TREATMENT_WEIGHTS, 1)[0])
            pre_treatment = PRE_TREATMENTS[pt_idx]
            pre_settling_hours = round(float(rng.uniform(0, 24) if "Settling" in pre_treatment or pre_treatment == "Gravity Separation" else rng.uniform(0, 8)), 1)

            # Operator
            op_pool = op_pools.get(fac, [f"OP-{fac}-01"])
            op_id   = op_pool[int(rng.integers(0, len(op_pool)))]
            op_exp  = op_exp_map.get(op_id, 5.0)

            # Yield calculation
            base_lo, base_hi = _BASE_YIELD_RANGES.get(subcat, (20, 50))
            base_yield = float(rng.uniform(base_lo, base_hi))

            # Modifiers
            exp_bonus    = op_exp * 0.3
            pt_bonus     = _PRE_TREATMENT_BONUS.get(pre_treatment, 0.0)
            equip_bonus  = (cond_score / 10.0) * 5.0
            visc_penalty = math.log10(max(1, visc40)) * 2.0
            emul_penalty = emulsion_pct * 0.2
            temp_bonus   = min((feed_temp - 40) * 0.1, 5.0)
            noise        = float(rng.normal(0, 3))

            final_yield = _clamp(
                base_yield + exp_bonus + pt_bonus + equip_bonus
                - visc_penalty - emul_penalty + temp_bonus + noise,
                15.0, 95.0
            )

            # Outputs
            total_oil_output = round(total_vol * oil_content * final_yield / 100, 2)
            total_water_output = round(total_vol * (1 - oil_content) * float(rng.uniform(0.7, 0.95)), 2)
            total_solids_kg    = round(total_vol * float(rng.uniform(0.005, 0.05)) * 1000, 1)

            # Processing duration hours
            equip_capacity = float(equip_row.get("capacity_m3_hr", 25)) if fac_equip else 25.0
            proc_hours = max(0.5, total_vol / max(equip_capacity, 1.0))
            ts_end = ts_start + timedelta(hours=proc_hours + pre_settling_hours)

            # Energy: 3-15 kWh/m³, higher for viscous and cold
            energy_factor = _clamp(3.0 + math.log10(max(1, visc40)) + max(0, (20 - feed_temp) * 0.1), 3.0, 15.0)
            energy_kwh = round(total_vol * energy_factor * float(rng.uniform(0.9, 1.1)), 1)

            # Costs
            chemicals_cost = round(total_vol * float(rng.uniform(2, 15)), 2)
            labor_hours    = round(proc_hours * float(rng.uniform(1.0, 2.0)), 1)
            labor_cost     = round(labor_hours * float(rng.uniform(35, 70)), 2)
            energy_cost    = round(energy_kwh * 0.12, 2)   # €0.12/kWh
            total_cost     = round(chemicals_cost + labor_cost + energy_cost, 2)

            # Revenue
            # Oil quality → price €400-600/mt, density ~900 kg/m³
            oil_price_per_mt = float(rng.uniform(400, 600))
            oil_mt = total_oil_output * 0.90  # ~900 kg/m³ → 0.9 mt/m³
            oil_revenue = round(oil_mt * oil_price_per_mt, 2)
            solids_revenue = round(total_solids_kg * float(rng.uniform(0.02, 0.08)), 2)
            net_margin = round(oil_revenue + solids_revenue - total_cost, 2)

            # Quality pass (85% pass rate)
            quality_pass = rng.random() < 0.85
            if not quality_pass:
                reasons = [
                    "High water content in oil output",
                    "Sulfur specification exceeded",
                    "Flash point below minimum",
                    "Viscosity out of spec",
                    "Sediment content too high",
                ]
                fail_reason = reasons[int(rng.integers(0, len(reasons)))]
            else:
                fail_reason = None

            # Target spec
            ts_idx = int(_weighted_choices(rng, TARGET_SPECS, TARGET_SPEC_WEIGHTS, 1)[0])
            target_spec = TARGET_SPECS[ts_idx]

            # Priority
            pri_idx = int(_weighted_choices(rng, BATCH_PRIORITIES, BATCH_PRIORITY_WEIGHTS, 1)[0])
            priority = BATCH_PRIORITIES[pri_idx]

            # Batch ID
            year = ts_start.year
            bkey = f"{fac}-{year}"
            batch_seq[bkey] = batch_seq.get(bkey, 0) + 1
            batch_id = f"BATCH-{fac}-{year}-{batch_seq[bkey]:05d}"

            rows.append({
                "batch_id":                 batch_id,
                "reception_id":             reception_ids_str,
                "start_timestamp":          ts_start,
                "end_timestamp":            ts_end,
                "facility_code":            fac,
                "operator_id":              op_id,
                "operator_experience_years": round(op_exp, 1),
                "separator_unit_id":        sep_unit_id,
                "pre_treatment":            pre_treatment,
                "pre_settling_hours":       pre_settling_hours,
                "target_oil_spec":          target_spec,
                "batch_priority":           priority,
                "total_volume_input_m3":    round(total_vol, 2),
                "total_oil_output_m3":      total_oil_output,
                "total_water_output_m3":    total_water_output,
                "total_solids_output_kg":   total_solids_kg,
                "oil_recovery_yield_pct":   round(final_yield, 2),
                "energy_consumed_kwh":      energy_kwh,
                "chemicals_cost_eur":       chemicals_cost,
                "labor_hours":              labor_hours,
                "total_processing_cost_eur": total_cost,
                "recovered_oil_revenue_eur": oil_revenue,
                "solid_fuel_revenue_eur":    solids_revenue,
                "net_margin_eur":            net_margin,
                "quality_pass":              quality_pass,
                "quality_failure_reason":    fail_reason,
            })

    df = pd.DataFrame(rows).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Function 4: generate_process_sensor_data
# ---------------------------------------------------------------------------

def generate_process_sensor_data(
    batch_df: pd.DataFrame,
    n_batches_sampled: int = 500,
) -> pd.DataFrame:
    """
    Sample n_batches_sampled batches and generate 1-minute sensor readings.
    """
    rng = _make_rng(RANDOM_SEED + 3)

    # Load equipment master for separator type lookup
    equip_path = os.path.join(
        os.path.dirname(__file__), "..", "reference", "equipment_master.csv"
    )
    try:
        equip_df = pd.read_csv(equip_path)
        equip_type_map = dict(zip(equip_df["unit_id"], equip_df["equipment_type"]))
        equip_hours_map = dict(zip(equip_df["unit_id"], equip_df["running_hours_since_overhaul"]))
    except FileNotFoundError:
        equip_type_map = {}
        equip_hours_map = {}

    # Sample batches
    n_sample = min(n_batches_sampled, len(batch_df))
    sampled = batch_df.sample(n=n_sample, random_state=RANDOM_SEED).reset_index(drop=True)

    rows = []

    for _, batch in sampled.iterrows():
        batch_id   = batch["batch_id"]
        fac        = batch["facility_code"]
        sep_id     = batch["separator_unit_id"]
        vol        = float(batch["total_volume_input_m3"])
        ts_start   = pd.Timestamp(batch["start_timestamp"])
        ts_end     = pd.Timestamp(batch["end_timestamp"])
        oil_yield  = float(batch["oil_recovery_yield_pct"]) / 100

        sep_type = equip_type_map.get(sep_id, "3-Phase Disc Stack Centrifuge")
        running_hrs = float(equip_hours_map.get(sep_id, 5000))

        # Duration in minutes
        duration_min = max(10, int((ts_end - ts_start).total_seconds() / 60))
        duration_min = min(duration_min, 2880)  # cap at 48 hours

        # Feed flow rate m³/hr
        feed_flow = _clamp(vol / (duration_min / 60), 1.0, 120.0)

        # Oil content from batch
        oil_content_frac = (batch["total_oil_output_m3"] / vol) if vol > 0 else 0.3
        water_content_frac = 1.0 - oil_content_frac - 0.02

        # Vibration baseline increases with running hours
        vib_baseline = 0.5 + (running_hrs / 50000) * 4.0

        # Sep-type-specific parameters
        if "Disc Stack" in sep_type:
            rpm_base       = float(rng.uniform(5000, 10000))
            diff_rpm       = None
            torque_base    = float(rng.uniform(200, 600))
        elif "Decanter" in sep_type:
            rpm_base       = float(rng.uniform(2000, 5000))
            diff_rpm       = float(rng.uniform(5, 80))
            torque_base    = float(rng.uniform(500, 2000))
        else:
            rpm_base       = 0.0
            diff_rpm       = None
            torque_base    = 0.0

        for min_idx in range(duration_min):
            ts = ts_start + timedelta(minutes=min_idx)
            noise = lambda pct=0.02: float(rng.normal(1.0, pct))

            # Feed temperature — slight warming during process
            feed_temp = float(batch["total_processing_cost_eur"])  # placeholder
            amb = temperature_for_location(ts.date(), fac)
            feed_temp_c = round((amb + 30) * noise(0.01), 1)

            # Feed flow with noise
            ff = round(feed_flow * noise(0.03), 2)

            # Centrifuge columns
            if rpm_base > 0:
                rpm   = round(rpm_base * noise(0.005), 0)
                torque = round(torque_base * noise(0.05), 1)
                vib   = round(
                    (vib_baseline + min_idx / duration_min * 0.5) * noise(0.10), 3
                )
                diff  = round(diff_rpm * noise(0.10), 2) if diff_rpm is not None else None
            else:
                rpm = diff = torque = vib = None

            # Oil discharge
            oil_flow = round(ff * oil_content_frac * oil_yield * noise(0.05), 3)
            oil_temp = round((feed_temp_c + float(rng.uniform(-2, 5))), 1)
            oil_bp   = round(float(rng.uniform(1.0, 6.0)) * noise(0.05), 2)

            # Water discharge
            wat_flow = round(ff * water_content_frac * noise(0.05), 3)
            wat_temp = round((feed_temp_c - float(rng.uniform(2, 8))), 1)

            # Solids discharge (interval-based)
            solids_interval = int(rng.integers(60, 1800))  # seconds between discharges
            solids_vol_l    = round(float(rng.uniform(5, 100)), 1)

            # Heating power
            heating_kw = round(float(rng.uniform(10, 80)) * noise(0.05), 1)

            # Chemical pump
            chem_ml_min = round(float(rng.uniform(50, 500)) * noise(0.10), 1)

            # Interface position (mm)
            interface_mm = round(float(rng.uniform(20, 150)) * noise(0.05), 1)

            # Turbidity
            turb_oil   = round(float(rng.uniform(0, 50)) * noise(0.10), 1)
            turb_water = round(float(rng.uniform(0, 100)) * noise(0.10), 1)

            # Online quality
            oil_in_water = round(float(rng.uniform(0, 15)) * noise(0.10), 2)
            water_in_oil = round(float(rng.uniform(0.1, 5.0)) * noise(0.10), 3)

            # Power & motor
            power_kw = round(float(rng.uniform(15, 150)) * noise(0.03), 1)
            motor_a  = round(power_kw / 0.38 * noise(0.02), 1)
            motor_temp = round(float(rng.uniform(45, 85)) * noise(0.02), 1)
            bearing_temp = round(float(rng.uniform(40, 75)) * noise(0.02), 1)

            rows.append({
                "reading_id":                str(uuid.uuid4()),
                "batch_id":                  batch_id,
                "timestamp":                 ts,
                "separator_unit_id":         sep_id,
                "separator_type":            sep_type,
                "feed_temperature_c":        feed_temp_c,
                "feed_flow_rate_m3_hr":      ff,
                "feed_pressure_bar":         round(float(rng.uniform(0.5, 4.0)) * noise(0.03), 2),
                "centrifuge_speed_rpm":      rpm,
                "centrifuge_differential_rpm": diff,
                "centrifuge_torque_nm":      torque,
                "centrifuge_vibration_mm_s": vib,
                "oil_discharge_temp_c":      oil_temp,
                "oil_discharge_flow_m3_hr":  oil_flow,
                "oil_backpressure_bar":      oil_bp,
                "water_discharge_temp_c":    wat_temp,
                "water_discharge_flow_m3_hr": wat_flow,
                "solids_discharge_interval_s": solids_interval,
                "solids_discharge_volume_l": solids_vol_l,
                "heating_power_kw":          heating_kw,
                "chemical_pump_rate_ml_min": chem_ml_min,
                "interface_position_mm":     interface_mm,
                "turbidity_oil_ntu":         turb_oil,
                "turbidity_water_ntu":       turb_water,
                "oil_in_water_ppm_online":   oil_in_water,
                "water_in_oil_pct_online":   water_in_oil,
                "power_consumption_kw":      power_kw,
                "motor_current_a":           motor_a,
                "motor_temperature_c":       motor_temp,
                "bearing_temperature_c":     bearing_temp,
            })

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Function 5: generate_weather_conditions
# ---------------------------------------------------------------------------

def generate_weather_conditions(n_years: float = 4.5) -> pd.DataFrame:
    """
    Hourly weather data for all 4 facilities × n_years ≈ 140,000 rows.
    """
    rng = _make_rng(RANDOM_SEED + 4)

    dt_start = datetime(DATE_START.year, DATE_START.month, DATE_START.day)
    end_dt   = dt_start + timedelta(days=int(n_years * 365.25))
    end_dt   = min(end_dt, datetime(DATE_END.year, DATE_END.month, DATE_END.day))

    rows = []

    # Generate hour-by-hour for each facility
    total_hours = int((end_dt - dt_start).total_seconds() / 3600)

    for fac in _FACILITY_CODES:
        ts = dt_start

        # Initialize AR(1) state for continuous simulation
        prev_wind = 5.0

        for h in range(total_hours):
            month = ts.month
            day_of_year = ts.timetuple().tm_yday
            hour = ts.hour

            # Ambient temperature: seasonal + diurnal + noise
            season_temp = temperature_for_location(ts.date(), fac)
            diurnal     = 3.0 * math.sin(math.pi * (hour - 6) / 12)
            temp_noise  = float(rng.normal(0, 1.0))
            temp_c      = round(season_temp + diurnal + temp_noise, 1)

            # Humidity: inverse of temperature roughly, but bounded
            if fac == "HAM":
                humidity_base = 75 - (temp_c - 10) * 0.5
            else:
                humidity_base = 65 - (temp_c - 17) * 0.4
            humidity = round(_clamp(humidity_base + float(rng.normal(0, 5)), 30, 100), 1)

            # Wind speed: AR(1) process with seasonal patterns
            if fac == "PIR":
                # Meltemi July-August: 15-25 m/s
                if month in (7, 8) and float(rng.random()) < 0.25:
                    wind_target = float(rng.uniform(15, 25))
                else:
                    wind_target = float(rng.uniform(2, 12))
            elif fac == "HAM":
                wind_target = float(rng.uniform(3, 15))
            elif fac == "GIB":
                # Levante wind: periodic strong easterlies
                if float(rng.random()) < 0.10:
                    wind_target = float(rng.uniform(12, 22))
                else:
                    wind_target = float(rng.uniform(2, 10))
            else:  # MLT
                wind_target = float(rng.uniform(2, 10))

            # AR(1) smoothing
            wind_speed = round(
                _clamp(0.7 * prev_wind + 0.3 * wind_target + float(rng.normal(0, 0.5)), 0, 30),
                1,
            )
            prev_wind = wind_speed

            # Precipitation
            if fac in ("PIR", "MLT"):
                # Dry summer
                if month in (6, 7, 8):
                    precip_prob = 0.03
                else:
                    precip_prob = 0.15
            elif fac == "HAM":
                precip_prob = 0.25  # rain year-round
            else:  # GIB
                if month in (6, 7, 8):
                    precip_prob = 0.05
                else:
                    precip_prob = 0.20

            if float(rng.random()) < precip_prob:
                precip = round(float(rng.exponential(2.0)), 2)
            else:
                precip = 0.0

            # Barometric pressure: ~1013 hPa ± seasonal
            pressure = round(
                1013 + float(rng.normal(0, 8)) - (month in (12, 1, 2)) * 3, 1
            )

            rows.append({
                "timestamp":           ts,
                "facility_code":       fac,
                "ambient_temperature_c": temp_c,
                "humidity_pct":          humidity,
                "wind_speed_ms":         wind_speed,
                "precipitation_mm":      precip,
                "barometric_pressure_hpa": pressure,
            })

            ts += timedelta(hours=1)

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Function 6: generate_all
# ---------------------------------------------------------------------------

def generate_all() -> dict:
    """
    Orchestrate all generators and return a dict of DataFrames.

    Returns
    -------
    dict with keys:
        "waste_reception_log"
        "laboratory_analysis"
        "processing_batch"
        "process_sensor_data"
        "weather_conditions"

    The returned dict also has a `save_all(output_dir)` helper method injected
    as an attribute of a thin wrapper class.
    """
    print("Generating waste reception log (n=15,000)...")
    reception_df = generate_waste_reception_log(n=15000)

    print("Generating laboratory analysis...")
    lab_df = generate_laboratory_analysis(reception_df)

    print("Generating processing batches...")
    batch_df = generate_processing_batch(reception_df, lab_df)

    print("Generating process sensor data (500 sampled batches)...")
    sensor_df = generate_process_sensor_data(batch_df, n_batches_sampled=500)

    print("Generating weather conditions (4.5 years)...")
    weather_df = generate_weather_conditions(n_years=4.5)

    result = {
        "waste_reception_log":  reception_df,
        "laboratory_analysis":  lab_df,
        "processing_batch":     batch_df,
        "process_sensor_data":  sensor_df,
        "weather_conditions":   weather_df,
    }

    # Attach save_all as a method on a simple Namespace-like class
    class _ResultDict(dict):
        def save_all(self, output_dir: str) -> None:
            """Save all DataFrames as CSV files to output_dir."""
            os.makedirs(output_dir, exist_ok=True)
            for name, df in self.items():
                path = os.path.join(output_dir, f"{name}.csv")
                df.to_csv(path, index=False)
                print(f"  Saved {path} ({len(df):,} rows)")

    out = _ResultDict(result)
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = generate_all()
    for name, df in result.items():
        print(f"{name}: {len(df)} rows, {len(df.columns)} cols")
