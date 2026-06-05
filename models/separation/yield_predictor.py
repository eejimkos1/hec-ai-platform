"""
Yield Predictor — XGBoost Regressor for oil_recovery_yield_pct.

Artifacts saved to models/trained/:
  yield_model.joblib
  yield_feature_importances.csv
  yield_evaluation_metrics.json
  yield_predictions_vs_actuals.csv
"""

import json
import os
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL = "oil_recovery_yield_pct"
EXCLUDE_COLS = {
    "batch_id",
    "oil_recovery_yield_pct",
    "quality_pass",
    "quality_failure_reason",
    "net_margin_eur",
}

MODEL_FILENAME = "yield_model.joblib"
IMPORTANCES_FILENAME = "yield_feature_importances.csv"
METRICS_FILENAME = "yield_evaluation_metrics.json"
PREDICTIONS_FILENAME = "yield_predictions_vs_actuals.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_dir(model_dir: str) -> pathlib.Path:
    base = pathlib.Path(__file__).resolve().parent.parent.parent
    p = pathlib.Path(model_dir)
    if not p.is_absolute():
        p = base / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_data(features_path: str):
    base = pathlib.Path(__file__).resolve().parent.parent.parent
    p = pathlib.Path(features_path)
    if not p.is_absolute():
        p = base / p
    df = pd.read_csv(p)
    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns, excluding targets and batch_id."""
    return [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_COLS
    ]


def _time_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Sort by batch_id (sequential) and split first 80% / last 20%."""
    df_sorted = df.sort_values("batch_id").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_frac)
    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]
    return train, test


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_and_save(
    features_path: str = "data/features/separation_features.csv",
    model_dir: str = "models/trained",
) -> dict:
    """
    Train XGBoost regressor for oil_recovery_yield_pct.

    Steps
    -----
    1. Load feature matrix.
    2. Time-based split: first 80% → train, last 20% → test.
    3. Feature columns = all numeric except targets and batch_id.
    4. Train XGBoost (n_estimators=200, max_depth=6, learning_rate=0.1).
    5. Evaluate: RMSE, MAE, R² on test set.
    6. Save artifacts.

    Returns
    -------
    dict with keys: rmse, mae, r2, train_size, test_size
    """
    df = _load_data(features_path)
    train_df, test_df = _time_split(df)

    feat_cols = _feature_columns(df)

    X_train = train_df[feat_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feat_cols]
    y_test = test_df[TARGET_COL]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",          # fast on CPU; handles NaN natively
        enable_categorical=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
    }

    # --- save artifacts ---
    out_dir = _resolve_dir(model_dir)

    joblib.dump({"model": model, "feature_columns": feat_cols}, out_dir / MODEL_FILENAME)

    importances = pd.DataFrame(
        {"feature": feat_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importances.to_csv(out_dir / IMPORTANCES_FILENAME, index=False)

    with open(out_dir / METRICS_FILENAME, "w") as fh:
        json.dump(metrics, fh, indent=2)

    preds_df = pd.DataFrame(
        {
            "batch_id": test_df["batch_id"].values,
            "actual": y_test.values,
            "predicted": y_pred,
            "residual": y_test.values - y_pred,
        }
    )
    preds_df.to_csv(out_dir / PREDICTIONS_FILENAME, index=False)

    print(
        f"[yield_predictor] R²={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}  "
        f"train={len(train_df):,}  test={len(test_df):,}"
    )
    return metrics


def predict(
    batch_features: dict,
    model_dir: str = "models/trained",
) -> dict:
    """
    Predict yield for a single batch.

    Parameters
    ----------
    batch_features : dict
        Key-value pairs for the feature columns. Missing values are filled with NaN
        (XGBoost handles them natively).

    Returns
    -------
    dict with keys: predicted_yield (float), confidence_interval (low, high)
    """
    out_dir = _resolve_dir(model_dir)
    # joblib.load is safe here: the file is written by train_and_save() in this
    # module to a controlled local directory (models/trained/).  It is never
    # loaded from an external or user-supplied path.
    artifact = joblib.load(out_dir / MODEL_FILENAME)
    model: XGBRegressor = artifact["model"]
    feat_cols: list[str] = artifact["feature_columns"]

    row = {col: batch_features.get(col, np.nan) for col in feat_cols}
    X = pd.DataFrame([row])

    y_hat = float(model.predict(X)[0])

    # Approximate 90 % interval using residual std from training predictions.
    # Stored in metrics file if available, else use a fixed ±10 pp fallback.
    metrics_path = out_dir / METRICS_FILENAME
    if metrics_path.exists():
        with open(metrics_path) as fh:
            m = json.load(fh)
        half_width = 1.645 * m.get("rmse", 5.0)
    else:
        half_width = 1.645 * 5.0

    return {
        "predicted_yield": round(y_hat, 2),
        "confidence_interval": (
            round(max(0.0, y_hat - half_width), 2),
            round(min(100.0, y_hat + half_width), 2),
        ),
    }


def get_feature_importances(
    top_n: int = 20,
    model_dir: str = "models/trained",
) -> pd.DataFrame:
    """
    Load and return top *top_n* feature importances.

    Returns
    -------
    pd.DataFrame with columns [feature, importance] sorted descending.
    """
    out_dir = _resolve_dir(model_dir)
    # joblib.load is safe here: the file is written by train_and_save() in this
    # module to a controlled local directory (models/trained/).  It is never
    # loaded from an external or user-supplied path.
    df = pd.read_csv(out_dir / IMPORTANCES_FILENAME)
    return df.head(top_n).reset_index(drop=True)
