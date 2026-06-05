"""
Parameter Optimizer — find optimal process parameters to maximise yield.

Uses the trained XGBoost yield predictor (yield_predictor.py) as the
surrogate objective function and scipy.optimize.minimize (L-BFGS-B) to
search the bounded parameter space.
"""

import pathlib
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# joblib is used here only to load artifacts written by yield_predictor.train_and_save()
# into the controlled local models/trained/ directory — not from any external source.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TUNABLE_PARAMS = {
    "feed_flow_rate_m3_hr":       (5.0,     80.0),
    "feed_temperature_c":         (40.0,    95.0),
    "centrifuge_speed_rpm":       (3000.0,  10000.0),
    "chemical_pump_rate_ml_min":  (0.0,     400.0),
    # The dataset column is residence_time_actual (seconds); bounds in seconds
    # to match the feature space (10–120 min → 600–7200 s).
    "residence_time_actual":      (600.0,   7200.0),
    "capacity_utilization":       (0.3,     0.95),
}

# Historical average operating conditions (from training data) used as the
# baseline for yield improvement calculations.  Values sourced from
# separation_features.csv column means so the delta reflects real-world gain.
BASELINE_PARAMS = {
    "feed_flow_rate_m3_hr":      19.23,
    "feed_temperature_c":        49.81,
    "centrifuge_speed_rpm":      6977.72,
    "chemical_pump_rate_ml_min": 490.32,
    "residence_time_actual":     899.65,
    "capacity_utilization":      0.47,
}

# Approximate EUR revenue per percentage-point of yield improvement.
# Derived from typical HEC batch economics; can be overridden via constraints.
EUR_PER_YIELD_PCT = 1_500.0

MODEL_FILENAME = "yield_model.joblib"
METRICS_FILENAME = "yield_evaluation_metrics.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_dir(model_dir: str) -> pathlib.Path:
    base = pathlib.Path(__file__).resolve().parent.parent.parent
    p = pathlib.Path(model_dir)
    if not p.is_absolute():
        p = base / p
    return p


def _load_model(model_dir: str = "models/trained"):
    """Load the trained yield predictor artifact (model + feature columns)."""
    out_dir = _resolve_dir(model_dir)
    # Safe: loading only from the controlled local models/trained/ directory
    # written by yield_predictor.train_and_save().
    artifact = joblib.load(out_dir / MODEL_FILENAME)
    return artifact["model"], artifact["feature_columns"]


def _build_feature_row(
    batch_characteristics: dict,
    tunable_values: dict,
    feat_cols: list[str],
) -> pd.DataFrame:
    """
    Merge fixed batch characteristics with the current tunable parameter
    values and fill anything else with NaN (XGBoost handles it natively).
    """
    row: dict[str, Any] = {}
    for col in feat_cols:
        if col in tunable_values:
            row[col] = tunable_values[col]
        elif col in batch_characteristics:
            row[col] = batch_characteristics[col]
        else:
            row[col] = np.nan
    return pd.DataFrame([row])


def _predict_yield(model, feat_cols, batch_characteristics, tunable_values) -> float:
    X = _build_feature_row(batch_characteristics, tunable_values, feat_cols)
    return float(model.predict(X)[0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize_parameters(
    batch_characteristics: dict,
    constraints: dict | None = None,
    model_dir: str = "models/trained",
) -> dict:
    """
    Find process parameters that maximise predicted oil recovery yield.

    Parameters
    ----------
    batch_characteristics : dict
        Fixed feedstock properties from lab analysis, e.g.
        {'viscosity_40c_cst': 500, 'oil_content_pct': 45, ...}
    constraints : dict, optional
        Override default safety / operational bounds.  Supported keys:
          max_temperature_c  (default: flash_point_c - 10 if flash_point_c given,
                              else 95)
          max_capacity_utilization (default: 0.95)
          eur_per_yield_pct  (default: 1 500)

    Returns
    -------
    dict:
      optimal_parameters          {param: value}
      predicted_yield             float  (%)
      yield_improvement_vs_baseline  float  (percentage points)
      estimated_revenue_impact_eur   float
    """
    model, feat_cols = _load_model(model_dir)
    constraints = constraints or {}

    # --- build bounds, honouring safety constraints -------------------------
    bounds_list = []
    param_names = list(TUNABLE_PARAMS.keys())

    # Safety: temperature must stay at least 10 °C below flash point
    flash_point = batch_characteristics.get("flash_point_c")
    max_temp = constraints.get(
        "max_temperature_c",
        (flash_point - 10.0) if flash_point is not None else 95.0,
    )
    max_temp = min(max_temp, TUNABLE_PARAMS["feed_temperature_c"][1])

    max_cap = constraints.get(
        "max_capacity_utilization",
        TUNABLE_PARAMS["capacity_utilization"][1],
    )
    max_cap = min(max_cap, 0.95)

    eur_per_pct = constraints.get("eur_per_yield_pct", EUR_PER_YIELD_PCT)

    for param in param_names:
        lo, hi = TUNABLE_PARAMS[param]
        if param == "feed_temperature_c":
            hi = max_temp
        elif param == "capacity_utilization":
            hi = max_cap
        bounds_list.append((lo, hi))

    # --- objective: negative yield (minimiser → maximise yield) ------------
    def neg_yield(x: np.ndarray) -> float:
        tunable = dict(zip(param_names, x))
        return -_predict_yield(model, feat_cols, batch_characteristics, tunable)

    # Starting point: mid-point of adjusted bounds
    x0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds_list])

    result = minimize(
        neg_yield,
        x0,
        method="L-BFGS-B",
        bounds=bounds_list,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    optimal_params = {
        param: round(float(val), 4)
        for param, val in zip(param_names, result.x)
    }
    predicted_yield = round(-float(result.fun), 2)

    # --- baseline: average / mid-point settings -----------------------------
    baseline_yield = _predict_yield(
        model, feat_cols, batch_characteristics, BASELINE_PARAMS
    )
    improvement = round(predicted_yield - baseline_yield, 2)
    revenue_impact = round(improvement * eur_per_pct, 2)

    return {
        "optimal_parameters": optimal_params,
        "predicted_yield": predicted_yield,
        "yield_improvement_vs_baseline": improvement,
        "estimated_revenue_impact_eur": revenue_impact,
    }


def what_if_analysis(
    batch_characteristics: dict,
    parameter_overrides: dict,
    model_dir: str = "models/trained",
) -> dict:
    """
    Predict yield when the user manually sets one or more process parameters.

    Parameters
    ----------
    batch_characteristics : dict
        Fixed feedstock properties.
    parameter_overrides : dict
        The parameter(s) the user wants to test, e.g. {'feed_temperature_c': 75}.

    Returns
    -------
    dict:
      predicted_yield     float
      baseline_yield      float  (mid-point settings)
      delta_yield         float  (override − baseline)
      parameters_used     {param: value}   (full set of tunable params applied)
    """
    model, feat_cols = _load_model(model_dir)

    # Merge baseline with user overrides
    params_used = {**BASELINE_PARAMS, **parameter_overrides}

    predicted_yield = round(
        _predict_yield(model, feat_cols, batch_characteristics, params_used), 2
    )
    baseline_yield = round(
        _predict_yield(model, feat_cols, batch_characteristics, BASELINE_PARAMS), 2
    )

    return {
        "predicted_yield": predicted_yield,
        "baseline_yield": baseline_yield,
        "delta_yield": round(predicted_yield - baseline_yield, 2),
        "parameters_used": {k: round(v, 4) for k, v in params_used.items()},
    }
