"""
profitability_model.py
HEC AI Platform - Task 8: Model 2C - XGBoost Voyage Profitability Model

Functions:
  train_and_save        - train XGBoost on voyage_history.csv, save artefacts
  predict_profitability - predict margin + class + key drivers for a voyage
"""

from __future__ import annotations

import json
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
TARGET_COL = "voyage_margin_eur"
DATE_COL = "departure_time"

FEATURE_COLS_BASE = [
    "distance_nm",
    "sea_time_hrs",
    "port_time_hrs",
    "cargo_loaded_m3",
    "fuel_consumed_mt",
    "collections_count",
    "avg_collection_time_hrs",
    "weather_delays_hrs",
    # engineered below:
    "vessel_class_enc",
    "departure_month",
    "is_winter",
]

# Profitability class thresholds (EUR margin)
HIGH_THRESHOLD   = 15_000
MEDIUM_THRESHOLD =  5_000
# < 0 -> Negative, else Low


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


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


def _margin_class(margin: float) -> str:
    if margin >= HIGH_THRESHOLD:
        return "High"
    if margin >= MEDIUM_THRESHOLD:
        return "Medium"
    if margin >= 0:
        return "Low"
    return "Negative"


def _encode_vessel_class(series: pd.Series) -> pd.Series:
    """Ordinal encode vessel class by typical size."""
    mapping = {
        "Coastal Small":  0,
        "Coastal Medium": 1,
        "Regional":       2,
        "Offshore Large": 3,
    }
    return series.map(mapping).fillna(1).astype(int)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_and_save(
    voyage_path: str = "data/raw/voyage_history.csv",
    model_dir: str = "models/trained",
) -> dict:
    """
    Train XGBoost regressor on voyage history to predict voyage_margin_eur.

    Steps
    -----
    1. Load voyage history.
    2. Target: voyage_margin_eur (drop rows where target is NaN).
    3. Engineer features: vessel_class_enc, departure_month, is_winter.
    4. Time-based 80/20 split on departure_time.
    5. Train XGBoost (n_estimators=150, max_depth=5).
    6. Evaluate: RMSE, MAE, R2.
    7. Save: profitability_model.joblib, metrics.json, feature_importances.csv.

    Returns
    -------
    dict with keys: rmse, mae, r2, train_rows, test_rows
    """
    import joblib
    from xgboost import XGBRegressor

    mdir = _model_dir_path(model_dir)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    vpath = Path(voyage_path) if Path(voyage_path).is_absolute() else _REPO_ROOT / voyage_path
    print(f"Loading voyage history from {vpath} ...")
    df = pd.read_csv(vpath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # ------------------------------------------------------------------
    # 2. Drop rows with missing target
    # ------------------------------------------------------------------
    df = df.dropna(subset=[TARGET_COL]).copy()
    print(f"  {len(df):,} voyages after dropping NaN target")

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    df["vessel_class_enc"] = _encode_vessel_class(df["vessel_class"])
    df["departure_month"]  = df[DATE_COL].dt.month
    df["is_winter"]        = df["departure_month"].isin([11, 12, 1, 2, 3]).astype(int)

    # Ensure all base columns exist (fill missing with 0)
    for col in FEATURE_COLS_BASE:
        if col not in df.columns:
            df[col] = 0.0

    # ------------------------------------------------------------------
    # 4. Time-based 80/20 split
    # ------------------------------------------------------------------
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    print(f"  Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

    X_train = train_df[FEATURE_COLS_BASE].values
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[FEATURE_COLS_BASE].values
    y_test  = test_df[TARGET_COL].values

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("Training XGBoost profitability model ...")
    model = XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    rmse_val = _rmse(y_test, y_pred)
    mae_val  = _mae(y_test, y_pred)
    r2_val   = _r2(y_test, y_pred)

    print(f"  RMSE: EUR{rmse_val:,.0f}  MAE: EUR{mae_val:,.0f}  R2: {r2_val:.3f}")

    # ------------------------------------------------------------------
    # 7. Save artefacts
    # ------------------------------------------------------------------
    model_path = mdir / "profitability_model.joblib"
    # Safe: we write and load this file entirely within our own pipeline;
    # it is never received from an untrusted external source.
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS_BASE}, model_path)
    print(f"  Saved model -> {model_path}")

    metrics = {
        "rmse": round(rmse_val, 2),
        "mae":  round(mae_val, 2),
        "r2":   round(r2_val, 3),
        "train_rows": int(len(train_df)),
        "test_rows":  int(len(test_df)),
    }
    metrics_path = mdir / "profitability_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importances
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature":    FEATURE_COLS_BASE,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    fi_path = mdir / "profitability_feature_importances.csv"
    fi_df.to_csv(fi_path, index=False)
    print(f"  Saved feature importances -> {fi_path}")

    print("Profitability model training complete.")
    return metrics


def predict_profitability(voyage_features: dict) -> dict:
    """
    Predict margin, class, and key drivers for a single voyage.

    Parameters
    ----------
    voyage_features : dict
        Keys matching any subset of FEATURE_COLS_BASE plus raw fields:
        vessel_class (str), departure_month (int, 1-12), etc.

    Returns
    -------
    dict:
        predicted_margin_eur   - float
        profitability_class    - 'High' | 'Medium' | 'Low' | 'Negative'
        key_drivers            - list of (feature, impact_eur) top-3 tuples
    """
    import joblib

    mdir = _model_dir_path("models/trained")
    model_path = mdir / "profitability_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_and_save() first."
        )

    # Safe: loading a file written by our own train_and_save() above.
    artifact = joblib.load(model_path)
    model: object = artifact["model"]
    feature_cols: list = artifact["feature_cols"]

    # Build feature dict with defaults
    feat = {c: 0.0 for c in feature_cols}

    # Direct field copies
    for c in feature_cols:
        if c in voyage_features:
            feat[c] = float(voyage_features[c])

    # Derived fields
    if "vessel_class" in voyage_features and "vessel_class_enc" not in voyage_features:
        feat["vessel_class_enc"] = float(
            _encode_vessel_class(pd.Series([voyage_features["vessel_class"]]))[0]
        )
    if "departure_month" not in voyage_features and "departure_time" in voyage_features:
        feat["departure_month"] = float(pd.to_datetime(voyage_features["departure_time"]).month)
    if "is_winter" not in voyage_features and "departure_month" in feat:
        feat["is_winter"] = float(int(feat["departure_month"]) in [11, 12, 1, 2, 3])

    X = np.array([[feat[c] for c in feature_cols]], dtype=float)
    predicted_margin = float(model.predict(X)[0])

    # Key drivers: contribution = feature_value × importance (scaled to EUR)
    importances = model.feature_importances_
    # Scale so the contributions sum to the predicted margin
    raw_contributions = np.array([feat[c] for c in feature_cols]) * importances
    scale = predicted_margin / (raw_contributions.sum() + 1e-9)
    contributions = raw_contributions * scale

    sorted_idx = np.argsort(np.abs(contributions))[::-1]
    key_drivers = [
        (feature_cols[i], round(float(contributions[i]), 2))
        for i in sorted_idx[:3]
    ]

    return {
        "predicted_margin_eur": round(predicted_margin, 2),
        "profitability_class":  _margin_class(predicted_margin),
        "key_drivers":          key_drivers,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    metrics = train_and_save()
    print(f"\nProfitability model - R2: {metrics['r2']:.3f}  RMSE: EUR{metrics['rmse']:,.0f}")

    # Sample prediction
    sample = {
        "distance_nm":            800.0,
        "sea_time_hrs":            66.0,
        "port_time_hrs":            5.0,
        "cargo_loaded_m3":        450.0,
        "fuel_consumed_mt":        14.0,
        "collections_count":        6,
        "avg_collection_time_hrs":  0.8,
        "weather_delays_hrs":       2.0,
        "vessel_class":         "Coastal Medium",
        "departure_month":          7,
    }
    result = predict_profitability(sample)
    print(f"\nSample prediction: {result}")
