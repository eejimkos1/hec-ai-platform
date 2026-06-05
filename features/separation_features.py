"""
separation_features.py

Transforms raw HEC source data into ML-ready feature matrices for the
petroleum waste separation use-case.

Main entry point:
    build_separation_feature_matrix(raw_data_dir, reference_dir, output_path)

Individual feature-group functions:
    compute_feedstock_features(batch_df, lab_df, reception_df)
    compute_process_features(batch_df, sensor_agg_df, equipment_df)
    compute_temporal_features(batch_df)
    compute_interaction_features(features_df)
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# mapping from reception mixing_during_transport → stability_factor
# The generator uses: None(nan), Low, Medium, High
# The spec names:     None=1.0, Mild=1.2, Moderate=1.5, Severe=2.0
# We map the generator levels to those factors:
_MIXING_STABILITY: dict = {
    "None":   1.0,
    "Low":    1.2,
    "Medium": 1.5,
    "High":   2.0,
}


def _safe_div(numerator: pd.Series, denominator: pd.Series,
              floor: float = 0.0) -> pd.Series:
    """Element-wise division guarded against zeros / NaN."""
    denom = denominator.copy().fillna(np.nan)
    denom = denom.where(denom.abs() > 1e-9, np.nan)
    return numerator / denom


# ---------------------------------------------------------------------------
# Category A — Feedstock Complexity Features
# ---------------------------------------------------------------------------

def compute_feedstock_features(
    batch_df: pd.DataFrame,
    lab_df: pd.DataFrame,
    reception_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Category-A feedstock complexity features.

    Joins: batch → primary reception → Inlet lab sample.

    Parameters
    ----------
    batch_df : pd.DataFrame
        processing_batch table (one row per batch).
    lab_df : pd.DataFrame
        laboratory_analysis table.
    reception_df : pd.DataFrame
        waste_reception_log table.

    Returns
    -------
    pd.DataFrame
        Index-aligned with batch_df; columns = feedstock feature names.
    """
    # --- 1. Extract primary reception_id (first token for multi-reception batches) ---
    primary_rec_id = batch_df["reception_id"].str.split(";").str[0]

    # --- 2. Inlet lab samples only ---
    inlet = (
        lab_df[lab_df["sample_point"] == "Inlet"]
        .drop_duplicates(subset=["reception_id"])
        .set_index("reception_id")
    )

    # --- 3. Reception lookup ---
    rec = reception_df.set_index("reception_id")

    # --- 4. Build combined feature frame aligned to batch_df ---
    n = len(batch_df)

    def _lookup_lab(col: str, default: float = np.nan) -> pd.Series:
        mapped = primary_rec_id.map(inlet[col]) if col in inlet.columns else pd.Series(np.nan, index=batch_df.index)
        return mapped.fillna(default).values

    def _lookup_rec(col: str, default=np.nan) -> pd.Series:
        mapped = primary_rec_id.map(rec[col]) if col in rec.columns else pd.Series(np.nan, index=batch_df.index)
        return mapped

    # Raw fields needed
    viscosity_40c    = pd.Series(_lookup_lab("viscosity_40c_cst",     100.0),  index=batch_df.index)
    emulsion_pct     = pd.Series(_lookup_lab("emulsion_layer_pct",      5.0),  index=batch_df.index)
    density          = pd.Series(_lookup_lab("density_15c_kg_m3",      900.0), index=batch_df.index)
    particle_d50     = pd.Series(_lookup_lab("particle_size_d50_micron", 10.0), index=batch_df.index)
    sulfur           = pd.Series(_lookup_lab("sulfur_total_pct",         1.0),  index=batch_df.index)
    oil_content      = pd.Series(_lookup_lab("oil_content_pct",         30.0), index=batch_df.index)
    vanadium         = pd.Series(_lookup_lab("vanadium_ppm",            0.0),  index=batch_df.index)
    nickel           = pd.Series(_lookup_lab("nickel_ppm",              0.0),  index=batch_df.index)
    iron             = pd.Series(_lookup_lab("iron_ppm",                0.0),  index=batch_df.index)
    sodium           = pd.Series(_lookup_lab("sodium_ppm",              0.0),  index=batch_df.index)
    alsi             = pd.Series(_lookup_lab("aluminum_silicon_ppm",    0.0),  index=batch_df.index)

    waste_temp_raw  = _lookup_rec("waste_temperature_c")
    mixing_raw      = _lookup_rec("mixing_during_transport")
    days_in_tank_raw = _lookup_rec("days_in_tank_before_collection")

    waste_temp   = pd.Series(waste_temp_raw.fillna(25.0).values.astype(float), index=batch_df.index)
    days_in_tank = pd.Series(days_in_tank_raw.fillna(0.0).values.astype(float), index=batch_df.index)

    # stability_factor from mixing_during_transport
    stability_factor = (
        mixing_raw
        .fillna("None")
        .map(_MIXING_STABILITY)
        .fillna(1.0)
        .astype(float)
    )
    stability_factor.index = batch_df.index

    # ---- Feature calculations ------------------------------------------------

    # viscosity_temperature_ratio = viscosity_40c / (waste_temperature + 273.15)
    viscosity_temperature_ratio = viscosity_40c / (waste_temp + 273.15)

    # emulsion_difficulty_score = emulsion_pct × (1 + viscosity_40c/1000) × stability_factor
    emulsion_difficulty_score = (
        emulsion_pct * (1.0 + viscosity_40c / 1000.0) * stability_factor
    )

    # oil_water_density_gap = 1000 - density_15c (bigger gap → easier separation)
    oil_water_density_gap = 1000.0 - density

    # solid_particle_settling_velocity (Stokes law, m/s)
    # v = (d/2)² × Δρ × g / (18 × μ)
    # d = particle_size_d50_micron / 1e6 metres
    # Δρ = density_15c - 800  [kg/m³]
    # μ = viscosity_40c × 1e-6 [Pa·s]  (cSt = mm²/s; dynamic ≈ cSt × density_kg_m3 × 1e-6, but spec uses cSt×1e-6 directly)
    d_m = particle_d50 / 1e6
    delta_rho = density - 800.0
    mu = viscosity_40c * 1e-6
    solid_particle_settling_velocity = (d_m ** 2) * delta_rho * 9.81 / (18.0 * mu.clip(lower=1e-9))

    # waste_age_degradation = log(days_in_tank + 1) × (1 + waste_temp/100)
    waste_age_degradation = np.log1p(days_in_tank) * (1.0 + waste_temp / 100.0)

    # sulfur_to_oil_ratio = sulfur / max(oil_content, 0.01)
    oil_content_safe = oil_content.clip(lower=0.01)
    sulfur_to_oil_ratio = sulfur / oil_content_safe

    # contamination_index = (V + Ni + Fe + Na) / 1000
    contamination_index = (vanadium + nickel + iron + sodium) / 1000.0

    # cat_fines_risk = 1 if aluminum_silicon_ppm > 60 else 0
    cat_fines_risk = (alsi > 60).astype(int)

    # Carry forward lab fields used by later feature groups
    result = pd.DataFrame(
        {
            # Raw fields for downstream use
            "viscosity_40c_cst":            viscosity_40c.values,
            "emulsion_layer_pct":           emulsion_pct.values,
            "density_15c_kg_m3":            density.values,
            "particle_size_d50_micron":     particle_d50.values,
            "sulfur_total_pct":             sulfur.values,
            "oil_content_pct":              oil_content.values,
            "water_content_pct":            pd.Series(_lookup_lab("water_content_pct", 50.0), index=batch_df.index).values,
            "solids_content_pct":           pd.Series(_lookup_lab("solids_content_pct", 5.0), index=batch_df.index).values,
            "pour_point_c":                 pd.Series(_lookup_lab("pour_point_c", 5.0), index=batch_df.index).values,
            # Computed features
            "viscosity_temperature_ratio":      viscosity_temperature_ratio.values,
            "emulsion_difficulty_score":        emulsion_difficulty_score.values,
            "oil_water_density_gap":            oil_water_density_gap.values,
            "solid_particle_settling_velocity": solid_particle_settling_velocity.values,
            "waste_age_degradation":            waste_age_degradation.values,
            "sulfur_to_oil_ratio":              sulfur_to_oil_ratio.values,
            "contamination_index":              contamination_index.values,
            "cat_fines_risk":                   cat_fines_risk.values,
        },
        index=batch_df.index,
    )
    return result


# ---------------------------------------------------------------------------
# Category B — Process Optimisation Features
# ---------------------------------------------------------------------------

def compute_process_features(
    batch_df: pd.DataFrame,
    sensor_agg_df: Optional[pd.DataFrame],
    equipment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Category-B process optimisation features.

    Parameters
    ----------
    batch_df : pd.DataFrame
        processing_batch table.
    sensor_agg_df : pd.DataFrame or None
        Per-batch mean of numeric sensor columns (batch_id as index).
        Pass None if sensor data is unavailable.
    equipment_df : pd.DataFrame
        equipment_master table.

    Returns
    -------
    pd.DataFrame
        Index-aligned with batch_df.
    """
    # ---- Equipment lookup by unit_id ----------------------------------------
    equip = equipment_df.set_index("unit_id")

    def _equip_col(col: str, default: float = np.nan) -> pd.Series:
        if col in equip.columns:
            return batch_df["separator_unit_id"].map(equip[col]).fillna(default)
        return pd.Series(default, index=batch_df.index)

    capacity_m3_hr = _equip_col("capacity_m3_hr", 25.0)
    bowl_radius_m  = _equip_col("bowl_radius_m",  0.33)

    # ---- Sensor data (mean per batch) ----------------------------------------
    has_sensor = sensor_agg_df is not None and len(sensor_agg_df) > 0

    def _sensor(col: str, default: float = np.nan) -> pd.Series:
        if has_sensor and col in sensor_agg_df.columns:
            return batch_df["batch_id"].map(sensor_agg_df[col])
        return pd.Series(np.nan, index=batch_df.index)

    sensor_feed_temp  = _sensor("feed_temperature_c")
    sensor_feed_flow  = _sensor("feed_flow_rate_m3_hr")
    sensor_rpm        = _sensor("centrifuge_speed_rpm")
    sensor_chem_pump  = _sensor("chemical_pump_rate_ml_min")

    # ---- Batch-level fallbacks -----------------------------------------------
    vol      = batch_df["total_volume_input_m3"].astype(float)
    energy   = batch_df["energy_consumed_kwh"].astype(float)
    duration = (
        (pd.to_datetime(batch_df["end_timestamp"]) - pd.to_datetime(batch_df["start_timestamp"]))
        .dt.total_seconds() / 3600.0
    ).clip(lower=0.01)

    # feed_flow_rate from batch if sensor missing
    feed_flow_batch = vol / duration  # m³/hr
    feed_flow = sensor_feed_flow.fillna(feed_flow_batch)

    # feed_temperature from batch pre_treatment / 50°C default if sensor missing
    feed_temp = sensor_feed_temp.fillna(50.0)

    # centrifuge_speed_rpm: 7000 rpm default if sensor missing
    rpm = sensor_rpm.fillna(7000.0)

    # chemical_pump_rate: estimate from chemicals_cost
    chem_pump_estimated = (
        batch_df["chemicals_cost_eur"].astype(float) / duration * 100.0
    ).clip(lower=50.0, upper=500.0)
    chem_pump = sensor_chem_pump.fillna(chem_pump_estimated)

    # emulsion_layer_pct for chemical_to_emulsion_ratio
    # We rely on feedstock features carrying this, but recompute from batch context
    # Use a placeholder of 5.0 if not available
    emulsion_pct = pd.Series(5.0, index=batch_df.index)

    # ---- Feature calculations ------------------------------------------------

    # specific_energy_input = energy / total_volume  (kWh/m³)
    specific_energy_input = _safe_div(energy, vol, floor=0.0)

    # g_force = (2π×rpm/60)² × bowl_radius / g
    omega = 2.0 * math.pi * rpm / 60.0
    g_force = (omega ** 2) * bowl_radius_m / 9.81

    # residence_time_actual = (vol / feed_flow_rate) × 60  (minutes)
    residence_time_actual = _safe_div(vol, feed_flow) * 60.0

    # chemical_to_emulsion_ratio = chem_pump_rate / max(emulsion_layer_pct, 1)
    emulsion_safe = emulsion_pct.clip(lower=1.0)
    chemical_to_emulsion_ratio = _safe_div(chem_pump, emulsion_safe)

    # temperature_vs_pour_point_margin = feed_temp - pour_point
    # pour_point comes from lab (feedstock features), use 5°C default here
    pour_point = pd.Series(5.0, index=batch_df.index)
    temperature_vs_pour_point_margin = feed_temp - pour_point

    # capacity_utilization = feed_flow / capacity_m3_hr
    capacity_utilization = _safe_div(feed_flow, capacity_m3_hr)

    result = pd.DataFrame(
        {
            # Raw fields for downstream use
            "feed_temperature_c":         feed_temp.values,
            "feed_flow_rate_m3_hr":       feed_flow.values,
            "centrifuge_speed_rpm":       rpm.values,
            "chemical_pump_rate_ml_min":  chem_pump.values,
            # Computed features
            "specific_energy_input":               specific_energy_input.values,
            "g_force":                             g_force.values,
            "residence_time_actual":               residence_time_actual.values,
            "chemical_to_emulsion_ratio":          chemical_to_emulsion_ratio.values,
            "temperature_vs_pour_point_margin":    temperature_vs_pour_point_margin.values,
            "capacity_utilization":                capacity_utilization.values,
        },
        index=batch_df.index,
    )
    return result


# ---------------------------------------------------------------------------
# Category C — Temporal / Contextual Features
# ---------------------------------------------------------------------------

def compute_temporal_features(
    batch_df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
    equipment_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute Category-C temporal and contextual features.

    Parameters
    ----------
    batch_df : pd.DataFrame
        processing_batch table, sorted by start_timestamp.
    weather_df : pd.DataFrame or None
        weather_conditions table for ambient_heating_delta.
    equipment_df : pd.DataFrame or None
        equipment_master for equipment_hours_since_service.

    Returns
    -------
    pd.DataFrame
        Index-aligned with batch_df.
    """
    df = batch_df.copy()
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"])
    df = df.sort_values("start_timestamp").copy()

    # Basic temporal extractions
    hour        = df["start_timestamp"].dt.hour
    dow         = df["start_timestamp"].dt.dayofweek  # 0=Mon
    month       = df["start_timestamp"].dt.month
    is_night    = hour.isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    is_weekend  = (dow >= 5).astype(int)

    # ---- Rolling yield 7d per facility ----------------------------------------
    df["_yield"] = df["oil_recovery_yield_pct"].astype(float)
    df["_date"]  = df["start_timestamp"].dt.floor("h")

    rolling_yield_7d_list  = []
    rolling_energy_7d_list = []

    # specific_energy placeholder until we have process features
    # We'll add it to batch_df context or use energy/volume directly
    df["_specific_energy"] = (
        df["energy_consumed_kwh"] / df["total_volume_input_m3"].replace(0, np.nan)
    )

    for fac, grp in df.groupby("facility_code", sort=False):
        grp = grp.sort_values("start_timestamp")
        # Set index to datetime for rolling time-based window
        grp_idx = grp.set_index("start_timestamp")

        ry7 = (
            grp_idx["_yield"]
            .rolling("7D", min_periods=1)
            .mean()
            .reset_index(drop=True)
        )
        re7 = (
            grp_idx["_specific_energy"]
            .rolling("7D", min_periods=1)
            .mean()
            .reset_index(drop=True)
        )
        rolling_yield_7d_list.append(
            pd.Series(ry7.values, index=grp.index, name="rolling_yield_7d")
        )
        rolling_energy_7d_list.append(
            pd.Series(re7.values, index=grp.index, name="rolling_energy_7d")
        )

    rolling_yield_7d  = pd.concat(rolling_yield_7d_list).reindex(df.index)
    rolling_energy_7d = pd.concat(rolling_energy_7d_list).reindex(df.index)

    # ---- Equipment hours since service ----------------------------------------
    if equipment_df is not None:
        equip = equipment_df.set_index("unit_id")
        if "running_hours_total" in equip.columns and "running_hours_since_overhaul" in equip.columns:
            hours_since_service = (
                df["separator_unit_id"]
                .map(equip["running_hours_total"] - equip["running_hours_since_overhaul"])
                .fillna(0.0)
            )
        else:
            hours_since_service = pd.Series(0.0, index=df.index)
    else:
        hours_since_service = pd.Series(0.0, index=df.index)

    # ---- Similar batch avg yield (same waste_subcategory, last 30d, same facility) ---
    # We join to batch_df which doesn't carry waste_subcategory; we need to add it.
    # Use operator-level rolling logic on what we have in batch_df.
    # waste_subcategory is not in batch_df directly — we'll mark as NaN if unavailable.
    # (It's set by the pipeline in build_separation_feature_matrix after joining reception)
    if "waste_subcategory" in df.columns:
        similar_batch_avg_yield = _rolling_similar_batch_yield(df)
    else:
        similar_batch_avg_yield = pd.Series(np.nan, index=df.index)

    # ---- Operator avg yield (last 90d) ----------------------------------------
    if "operator_id" in df.columns:
        op_avg_yield = _rolling_operator_yield(df)
    else:
        op_avg_yield = pd.Series(np.nan, index=df.index)

    # ---- Ambient heating delta ------------------------------------------------
    if weather_df is not None:
        ambient_heating_delta = _compute_ambient_heating_delta(df, weather_df)
    else:
        ambient_heating_delta = pd.Series(np.nan, index=df.index)

    result = pd.DataFrame(
        {
            "hour_of_day":                 hour.values,
            "day_of_week":                 dow.values,
            "month":                       month.values,
            "is_night_shift":              is_night.values,
            "is_weekend":                  is_weekend.values,
            "rolling_yield_7d":            rolling_yield_7d.values,
            "rolling_energy_7d":           rolling_energy_7d.values,
            "equipment_hours_since_service": hours_since_service.values,
            "similar_batch_avg_yield":     similar_batch_avg_yield.values,
            "operator_avg_yield":          op_avg_yield.values,
            "ambient_heating_delta":       ambient_heating_delta.values,
        },
        index=df.index,
    )
    # Re-align to original batch_df index order
    return result.reindex(batch_df.index)


def _rolling_similar_batch_yield(df: pd.DataFrame) -> pd.Series:
    """
    For each batch, compute mean yield of same waste_subcategory
    at same facility over the preceding 30 days.
    """
    out = pd.Series(np.nan, index=df.index)
    grouped = df.groupby(["facility_code", "waste_subcategory"], sort=False)
    for (fac, subcat), grp in grouped:
        grp = grp.sort_values("start_timestamp")
        ts  = grp["start_timestamp"]
        yld = grp["oil_recovery_yield_pct"].astype(float)
        vals = []
        for i, (idx, row_ts) in enumerate(ts.items()):
            window_start = row_ts - pd.Timedelta(days=30)
            past = yld[(ts < row_ts) & (ts >= window_start)]
            vals.append(past.mean() if len(past) > 0 else np.nan)
        out.loc[grp.index] = vals
    return out


def _rolling_operator_yield(df: pd.DataFrame) -> pd.Series:
    """
    For each batch, compute operator mean yield over the preceding 90 days.
    """
    out = pd.Series(np.nan, index=df.index)
    for op_id, grp in df.groupby("operator_id", sort=False):
        grp = grp.sort_values("start_timestamp")
        ts  = grp["start_timestamp"]
        yld = grp["oil_recovery_yield_pct"].astype(float)
        vals = []
        for i, (idx, row_ts) in enumerate(ts.items()):
            window_start = row_ts - pd.Timedelta(days=90)
            past = yld[(ts < row_ts) & (ts >= window_start)]
            vals.append(past.mean() if len(past) > 0 else np.nan)
        out.loc[grp.index] = vals
    return out


def _compute_ambient_heating_delta(
    batch_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute feed_temperature_c - ambient_temperature_c matched to nearest hour.
    Uses batch feed_temperature from batch_df (50°C default) minus weather ambient.
    """
    out = pd.Series(np.nan, index=batch_df.index)

    # Build a weather lookup: (facility, floored-hour) → ambient_temp
    w = weather_df.copy()
    w["ts_floor"] = pd.to_datetime(w["timestamp"]).dt.floor("h")
    w_idx = w.set_index(["facility_code", "ts_floor"])["ambient_temperature_c"]

    for idx, row in batch_df.iterrows():
        fac = row["facility_code"]
        ts  = pd.to_datetime(row["start_timestamp"]).floor("h")
        key = (fac, ts)
        if key in w_idx.index:
            amb = float(w_idx[key])
            # feed_temperature from batch context (process features fills this in)
            feed_temp = 50.0  # conservative default; overridden after merge
            out.loc[idx] = feed_temp - amb
        # else stays NaN
    return out


# ---------------------------------------------------------------------------
# Category D — Interaction Features
# ---------------------------------------------------------------------------

def compute_interaction_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Category-D interaction features from the combined feature matrix.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing all previously computed features (A + B + C).

    Returns
    -------
    pd.DataFrame
        The interaction features only, index-aligned to features_df.
    """
    def _col(name: str, default: float = 0.0) -> pd.Series:
        if name in features_df.columns:
            return features_df[name].fillna(default)
        return pd.Series(default, index=features_df.index)

    viscosity_x_flow_rate = (
        _col("viscosity_40c_cst") * _col("feed_flow_rate_m3_hr")
    )
    solids_x_rpm = (
        _col("solids_content_pct") * _col("centrifuge_speed_rpm")
    )
    water_x_temperature = (
        _col("water_content_pct") * (100.0 - _col("feed_temperature_c"))
    )
    sulfur_x_volume = (
        _col("sulfur_total_pct") * _col("total_volume_input_m3")
    )
    age_x_emulsion = (
        _col("waste_age_degradation") * _col("emulsion_difficulty_score")
    )

    return pd.DataFrame(
        {
            "viscosity_x_flow_rate": viscosity_x_flow_rate.values,
            "solids_x_rpm":          solids_x_rpm.values,
            "water_x_temperature":   water_x_temperature.values,
            "sulfur_x_volume":       sulfur_x_volume.values,
            "age_x_emulsion":        age_x_emulsion.values,
        },
        index=features_df.index,
    )


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def build_separation_feature_matrix(
    raw_data_dir: str = "data/raw",
    reference_dir: str = "data/reference",
    output_path: str = "data/features/separation_features.csv",
) -> pd.DataFrame:
    """
    Load raw data, engineer all features, and save the feature matrix.

    Parameters
    ----------
    raw_data_dir : str
        Directory containing raw CSV files.
    reference_dir : str
        Directory containing reference CSVs (equipment_master.csv).
    output_path : str
        Path to save the final feature matrix CSV.

    Returns
    -------
    pd.DataFrame
        Full feature matrix (~12,000 rows, one per batch).
    """
    logger.info("Loading raw data from %s", raw_data_dir)

    # ------------------------------------------------------------------ #
    # Step 1 – Generate raw CSVs if they don't exist                      #
    # ------------------------------------------------------------------ #
    raw_files = {
        "waste_reception_log": os.path.join(raw_data_dir, "waste_reception_log.csv"),
        "laboratory_analysis": os.path.join(raw_data_dir, "laboratory_analysis.csv"),
        "processing_batch":    os.path.join(raw_data_dir, "processing_batch.csv"),
        "process_sensor_data": os.path.join(raw_data_dir, "process_sensor_data.csv"),
        "weather_conditions":  os.path.join(raw_data_dir, "weather_conditions.csv"),
    }
    missing = [k for k, p in raw_files.items() if not os.path.exists(p)]
    if missing:
        logger.info("Raw CSVs missing (%s), generating...", missing)
        from data.generators.separation_data_generator import generate_all
        data = generate_all()
        os.makedirs(raw_data_dir, exist_ok=True)
        for name, df in data.items():
            df.to_csv(raw_files[name], index=False)

    # ------------------------------------------------------------------ #
    # Step 2 – Load all tables                                            #
    # ------------------------------------------------------------------ #
    reception_df = pd.read_csv(raw_files["waste_reception_log"])
    lab_df       = pd.read_csv(raw_files["laboratory_analysis"])
    batch_df     = pd.read_csv(raw_files["processing_batch"])
    sensor_df    = pd.read_csv(raw_files["process_sensor_data"])
    weather_df   = pd.read_csv(raw_files["weather_conditions"])

    equip_path = os.path.join(reference_dir, "equipment_master.csv")
    equipment_df = pd.read_csv(equip_path)

    logger.info(
        "Loaded: batch=%d, reception=%d, lab=%d, sensor=%d, weather=%d, equip=%d",
        len(batch_df), len(reception_df), len(lab_df),
        len(sensor_df), len(weather_df), len(equipment_df),
    )

    # ------------------------------------------------------------------ #
    # Step 3 – Join batch → primary reception → waste_subcategory         #
    # ------------------------------------------------------------------ #
    batch_df = batch_df.reset_index(drop=True)
    batch_df["primary_reception_id"] = batch_df["reception_id"].str.split(";").str[0]

    rec_cols = ["reception_id", "facility_code", "waste_subcategory",
                "waste_temperature_c", "mixing_during_transport",
                "days_in_tank_before_collection"]
    rec_slim = reception_df[rec_cols].rename(
        columns={"reception_id": "primary_reception_id",
                 "facility_code": "_rec_facility"}
    )
    batch_df = batch_df.merge(rec_slim, on="primary_reception_id", how="left")

    # ------------------------------------------------------------------ #
    # Step 4 – Aggregate sensor data per batch (mean of numeric columns)  #
    # ------------------------------------------------------------------ #
    numeric_sensor_cols = sensor_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude non-feature columns that happen to be numeric
    exclude_sensor = {"reading_id"}
    agg_cols = [c for c in numeric_sensor_cols if c not in exclude_sensor]

    sensor_agg = (
        sensor_df.groupby("batch_id")[agg_cols]
        .mean()
    )
    # sensor_agg is indexed by batch_id

    # ------------------------------------------------------------------ #
    # Step 5 – Merge weather: nearest hour match to batch start           #
    # ------------------------------------------------------------------ #
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
    batch_df["start_timestamp"] = pd.to_datetime(batch_df["start_timestamp"])

    weather_hourly = (
        weather_df
        .assign(ts_floor=lambda d: d["timestamp"].dt.floor("h"))
        .set_index(["facility_code", "ts_floor"])
    )

    # Vectorised weather merge using floored start_timestamp
    batch_df["_ts_floor"] = batch_df["start_timestamp"].dt.floor("h")
    batch_df["_fac_ts"]   = list(zip(batch_df["facility_code"], batch_df["_ts_floor"]))

    ambient_temps = []
    for fac, ts in batch_df["_fac_ts"]:
        key = (fac, ts)
        if key in weather_hourly.index:
            ambient_temps.append(float(weather_hourly.loc[key, "ambient_temperature_c"].iloc[0]
                                      if hasattr(weather_hourly.loc[key, "ambient_temperature_c"], '__len__')
                                      else weather_hourly.loc[key, "ambient_temperature_c"]))
        else:
            ambient_temps.append(np.nan)

    batch_df["ambient_temperature_c"] = ambient_temps
    batch_df.drop(columns=["_ts_floor", "_fac_ts"], inplace=True)

    logger.info("Weather merge complete. NaN ambient: %d",
                batch_df["ambient_temperature_c"].isna().sum())

    # ------------------------------------------------------------------ #
    # Step 6 – Compute feedstock features                                 #
    # ------------------------------------------------------------------ #
    logger.info("Computing feedstock features (Category A)...")
    feedstock_feats = compute_feedstock_features(batch_df, lab_df, reception_df)

    # ------------------------------------------------------------------ #
    # Step 7 – Compute process features                                   #
    # ------------------------------------------------------------------ #
    logger.info("Computing process features (Category B)...")
    process_feats = compute_process_features(batch_df, sensor_agg, equipment_df)

    # ------------------------------------------------------------------ #
    # Step 8 – Compute temporal features                                  #
    # ------------------------------------------------------------------ #
    logger.info("Computing temporal features (Category C)...")
    # Augment batch_df with feedstock/process raw fields for temporal use
    temporal_input = batch_df.copy()
    temporal_feats = compute_temporal_features(
        temporal_input, weather_df=weather_df, equipment_df=equipment_df
    )

    # ------------------------------------------------------------------ #
    # Step 9 – Build combined feature frame for interactions              #
    # ------------------------------------------------------------------ #
    combined = pd.concat(
        [feedstock_feats, process_feats, temporal_feats], axis=1
    )
    # Add total_volume_input_m3 so interaction features can reference it
    combined["total_volume_input_m3"] = batch_df["total_volume_input_m3"].values

    # Update ambient_heating_delta now that feed_temperature_c is known
    # feed_temp is in process_feats, ambient is on batch_df
    if "ambient_temperature_c" in batch_df.columns:
        combined["ambient_heating_delta"] = (
            process_feats["feed_temperature_c"].values
            - batch_df["ambient_temperature_c"].fillna(15.0).values
        )

    # ------------------------------------------------------------------ #
    # Step 10 – Compute interaction features                              #
    # ------------------------------------------------------------------ #
    logger.info("Computing interaction features (Category D)...")
    interaction_feats = compute_interaction_features(combined)

    # ------------------------------------------------------------------ #
    # Step 11 – Assemble final feature matrix                             #
    # ------------------------------------------------------------------ #
    target_cols = [
        "batch_id",
        "oil_recovery_yield_pct",
        "quality_pass",
        "quality_failure_reason",
        "net_margin_eur",
    ]
    targets = batch_df[target_cols].reset_index(drop=True)

    # Feature columns (all except raw pass-through fields and targets)
    feature_only_cols = [
        # --- Category A ---
        "viscosity_temperature_ratio",
        "emulsion_difficulty_score",
        "oil_water_density_gap",
        "solid_particle_settling_velocity",
        "waste_age_degradation",
        "sulfur_to_oil_ratio",
        "contamination_index",
        "cat_fines_risk",
        # --- raw fields kept for interaction / process use ---
        "viscosity_40c_cst",
        "emulsion_layer_pct",
        "density_15c_kg_m3",
        "particle_size_d50_micron",
        "sulfur_total_pct",
        "oil_content_pct",
        "water_content_pct",
        "solids_content_pct",
        "pour_point_c",
        # --- Category B ---
        "specific_energy_input",
        "g_force",
        "residence_time_actual",
        "chemical_to_emulsion_ratio",
        "temperature_vs_pour_point_margin",
        "capacity_utilization",
        "feed_temperature_c",
        "feed_flow_rate_m3_hr",
        "centrifuge_speed_rpm",
        "chemical_pump_rate_ml_min",
        # --- Category C ---
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
        # --- Category D ---
        "viscosity_x_flow_rate",
        "solids_x_rpm",
        "water_x_temperature",
        "sulfur_x_volume",
        "age_x_emulsion",
    ]

    feature_df = pd.concat(
        [
            combined.reset_index(drop=True),
            interaction_feats.reset_index(drop=True),
        ],
        axis=1,
    )

    # Retain only feature columns that exist
    keep = [c for c in feature_only_cols if c in feature_df.columns]
    feature_df = feature_df[keep].reset_index(drop=True)

    final_df = pd.concat([targets, feature_df], axis=1)

    logger.info("Feature matrix shape: %s", final_df.shape)
    logger.info("Total nulls: %d", final_df.isnull().sum().sum())

    # ------------------------------------------------------------------ #
    # Step 12 – Save                                                      #
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info("Saved feature matrix to %s", output_path)

    return final_df
