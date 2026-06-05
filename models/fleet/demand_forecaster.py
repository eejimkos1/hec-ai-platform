"""
demand_forecaster.py
HEC AI Platform - Task 8: Model 2A - XGBoost Waste Volume Demand Forecaster

Functions:
  train_and_save          - train XGBoost on fleet feature matrix, save artefacts
  forecast_port           - generate N-day ahead forecast for a single port
  get_forecast_accuracy_by_port - return per-port metrics from saved metrics file
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_CUTOFF = "2025-07-01"
TARGET_COL = "waste_volume_collected_m3"
PORT_COL = "port_code"
DATE_COL = "date"

# Columns to always exclude from features
_EXCLUDE_COLS = {
    "demand_id",
    "date",
    "port_code",
    "port_name",
    "port_country",
    "vessel_calls_by_type_container",
    "vessel_calls_by_type_tanker",
    "vessel_calls_by_type_bulk",
    "vessel_calls_by_type_cruise",
    "vessel_calls_by_type_other",
    "waste_bilge_m3",
    "waste_sludge_m3",
    "waste_slops_m3",
    "waste_other_m3",
    "weather_zone",
    # target & leak columns
    "waste_volume_collected_m3",
    "waste_7d_rolling",
    "waste_30d_rolling",
    "expected_revenue_per_voyage",
    "expected_margin",
}

LAG_DAYS = [7, 14, 30]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def _model_dir_path(model_dir: str) -> Path:
    p = Path(model_dir)
    if not p.is_absolute():
        p = _REPO_ROOT / model_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_and_save(
    features_path: str = "data/features/fleet_features.csv",
    model_dir: str = "models/trained",
) -> dict:
    """
    Train XGBoost regressor on fleet feature matrix.

    Steps
    -----
    1. Load fleet feature matrix.
    2. Target: waste_volume_collected_m3.
    3. Create lag features: waste_volume lag 7, 14, 30 days (per port).
    4. Time-based split: train < 2025-07-01, test >= 2025-07-01.
    5. Feature columns: all computed features + lag features.
    6. Train XGBoost (n_estimators=200, max_depth=6, learning_rate=0.08).
    7. Evaluate: MAPE, RMSE, R2 on test set overall and per port.
    8. Save artefacts: demand_forecaster.joblib, metrics.json,
       port_metrics.csv, predictions.csv.

    Returns
    -------
    dict with keys: mape, rmse, r2, port_metrics (list of dicts)
    """
    import joblib
    from xgboost import XGBRegressor

    mdir = _model_dir_path(model_dir)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    feat_path = Path(features_path) if Path(features_path).is_absolute() else _REPO_ROOT / features_path
    if not feat_path.exists():
        print("Fleet feature matrix not found - generating ...")
        from features.fleet_features import build_fleet_feature_matrix
        build_fleet_feature_matrix()

    print(f"Loading fleet features from {feat_path} ...")
    df = pd.read_csv(feat_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([PORT_COL, DATE_COL]).reset_index(drop=True)

    # Drop rows where target is NaN
    df = df.dropna(subset=[TARGET_COL])

    # ------------------------------------------------------------------
    # 3. Lag features per port
    # ------------------------------------------------------------------
    print("Creating lag features ...")
    for lag in LAG_DAYS:
        col_name = f"waste_volume_lag_{lag}d"
        df[col_name] = (
            df.groupby(PORT_COL)[TARGET_COL]
            .shift(lag)
        )

    # ------------------------------------------------------------------
    # 4. Time-based split
    # ------------------------------------------------------------------
    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    train_df = df[df[DATE_COL] < cutoff].copy()
    test_df = df[df[DATE_COL] >= cutoff].copy()

    print(f"Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

    if test_df.empty:
        raise ValueError(
            f"No test data found after cutoff {TRAIN_CUTOFF}. "
            "Check date range in fleet_features.csv."
        )

    # ------------------------------------------------------------------
    # 5. Feature columns
    # ------------------------------------------------------------------
    lag_cols = [f"waste_volume_lag_{lag}d" for lag in LAG_DAYS]
    base_feature_cols = [
        c for c in df.columns
        if c not in _EXCLUDE_COLS and c not in lag_cols
    ]
    # Encode any remaining string columns
    cat_cols = [c for c in base_feature_cols if df[c].dtype == object]
    for c in cat_cols:
        df[c] = df[c].astype("category").cat.codes
        train_df[c] = train_df[c].astype("category").cat.codes
        test_df[c] = test_df[c].astype("category").cat.codes

    feature_cols = base_feature_cols + lag_cols

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].values

    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].values

    # ------------------------------------------------------------------
    # 6. Train XGBoost
    # ------------------------------------------------------------------
    print("Training XGBoost demand forecaster ...")
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # ------------------------------------------------------------------
    # 7. Evaluate overall
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    overall_mape = _mape(y_test, y_pred)
    overall_rmse = _rmse(y_test, y_pred)
    overall_r2 = _r2(y_test, y_pred)

    print(f"  Overall - MAPE: {overall_mape:.1f}%  RMSE: {overall_rmse:.1f}  R2: {overall_r2:.3f}")

    # Per-port metrics
    test_with_pred = test_df[[PORT_COL, DATE_COL, TARGET_COL]].copy()
    test_with_pred["predicted_volume_m3"] = y_pred

    port_metrics_records = []
    for port, grp in test_with_pred.groupby(PORT_COL):
        yt = grp[TARGET_COL].values
        yp = grp["predicted_volume_m3"].values
        port_metrics_records.append({
            "port_code": port,
            "mape": round(_mape(yt, yp), 2),
            "rmse": round(_rmse(yt, yp), 2),
            "r2": round(_r2(yt, yp), 3),
            "n_samples": len(yt),
        })

    # ------------------------------------------------------------------
    # 8. Save artefacts
    # ------------------------------------------------------------------
    model_path = mdir / "demand_forecaster.joblib"
    # Safe: we write this file ourselves in a controlled training pipeline.
    # It is never loaded from an untrusted external source.
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"  Saved model -> {model_path}")

    metrics = {
        "mape": round(overall_mape, 2),
        "rmse": round(overall_rmse, 2),
        "r2": round(overall_r2, 3),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_cutoff": TRAIN_CUTOFF,
        "port_metrics": port_metrics_records,
    }

    metrics_path = mdir / "demand_forecaster_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics -> {metrics_path}")

    port_metrics_df = pd.DataFrame(port_metrics_records)
    port_metrics_path = mdir / "demand_forecaster_port_metrics.csv"
    port_metrics_df.to_csv(port_metrics_path, index=False)

    predictions_path = mdir / "demand_forecaster_predictions.csv"
    test_with_pred.to_csv(predictions_path, index=False)

    print("Demand forecaster training complete.")
    return metrics


def forecast_port(port_code: str, horizon_days: int = 7) -> pd.DataFrame:
    """
    Generate a rolling horizon forecast for a single port.

    Uses the last available row in fleet_features.csv as the seed, then
    generates N future dates using the trained model by propagating lag
    features forward.

    Returns
    -------
    pd.DataFrame with columns: date, predicted_volume_m3, lower_bound, upper_bound
    """
    import joblib

    mdir = _model_dir_path("models/trained")
    model_path = mdir / "demand_forecaster.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_and_save() first."
        )

    # Safe: loading a file written by our own training pipeline above -
    # never loaded from an untrusted external source.
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    feat_path = _REPO_ROOT / "data/features/fleet_features.csv"
    df = pd.read_csv(feat_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([PORT_COL, DATE_COL])

    port_df = df[df[PORT_COL] == port_code].copy()
    if port_df.empty:
        raise ValueError(f"Port '{port_code}' not found in fleet_features.csv")

    # Add lag columns based on historical data
    for lag in LAG_DAYS:
        col_name = f"waste_volume_lag_{lag}d"
        port_df[col_name] = port_df[TARGET_COL].shift(lag)

    # Encode categorical columns
    for c in feature_cols:
        if c in port_df.columns and port_df[c].dtype == object:
            port_df[c] = port_df[c].astype("category").cat.codes

    last_date = port_df[DATE_COL].max()
    last_known = port_df[TARGET_COL].values.tolist()

    records = []
    for step in range(1, horizon_days + 1):
        future_date = last_date + pd.Timedelta(days=step)

        # Build a row using the last known feature row, updating lags
        seed_row = port_df.iloc[-1:].copy()
        seed_row[DATE_COL] = future_date

        for lag in LAG_DAYS:
            col_name = f"waste_volume_lag_{lag}d"
            idx = -lag  # look back lag steps in last_known
            if abs(idx) <= len(last_known):
                seed_row[col_name] = last_known[idx]
            else:
                seed_row[col_name] = float("nan")

        # Ensure all feature cols exist
        for c in feature_cols:
            if c not in seed_row.columns:
                seed_row[c] = 0.0

        X_pred = seed_row[feature_cols].values
        pred_vol = float(model.predict(X_pred)[0])
        pred_vol = max(0.0, pred_vol)

        records.append({
            "date": future_date,
            "predicted_volume_m3": round(pred_vol, 1),
            "lower_bound": round(pred_vol * 0.80, 1),
            "upper_bound": round(pred_vol * 1.20, 1),
        })

        last_known.append(pred_vol)

    return pd.DataFrame(records)


def get_forecast_accuracy_by_port() -> pd.DataFrame:
    """
    Return per-port MAPE and RMSE from saved metrics.

    Returns
    -------
    pd.DataFrame with columns: port_code, mape, rmse, r2, n_samples
    """
    mdir = _model_dir_path("models/trained")
    metrics_path = mdir / "demand_forecaster_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics not found at {metrics_path}. Run train_and_save() first."
        )
    with open(metrics_path) as f:
        metrics = json.load(f)
    return pd.DataFrame(metrics.get("port_metrics", []))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    metrics = train_and_save()
    print(f"\nDemand model - MAPE: {metrics['mape']:.1f}%  R2: {metrics['r2']:.3f}")
    acc = get_forecast_accuracy_by_port()
    print("\nPer-port accuracy:")
    print(acc.to_string(index=False))
