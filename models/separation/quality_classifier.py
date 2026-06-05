"""
Quality Classifier — Random Forest models for quality_pass and failure reason.

Artifacts saved to models/trained/:
  quality_classifier.joblib          (binary pass/fail)
  failure_reason_classifier.joblib   (multi-class, trained on failed batches only)
  classification_report.json
"""

import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_PASS = "quality_pass"
TARGET_REASON = "quality_failure_reason"
EXCLUDE_COLS = {
    "batch_id",
    "oil_recovery_yield_pct",
    "quality_pass",
    "quality_failure_reason",
    "net_margin_eur",
}

PASS_MODEL_FILENAME = "quality_classifier.joblib"
REASON_MODEL_FILENAME = "failure_reason_classifier.joblib"
REPORT_FILENAME = "classification_report.json"


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


def _load_data(features_path: str) -> pd.DataFrame:
    base = pathlib.Path(__file__).resolve().parent.parent.parent
    p = pathlib.Path(features_path)
    if not p.is_absolute():
        p = base / p
    return pd.read_csv(p)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_COLS
    ]


def _time_split(df: pd.DataFrame, train_frac: float = 0.8):
    df_sorted = df.sort_values("batch_id").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_frac)
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def _make_rf_pipeline(n_estimators: int = 150, random_state: int = 42) -> Pipeline:
    """RandomForest with median imputation for NaN features."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=None,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_and_save(
    features_path: str = "data/features/separation_features.csv",
    model_dir: str = "models/trained",
) -> dict:
    """
    Train two Random Forest classifiers.

    Model A — binary classifier for quality_pass (True/False).
    Model B — multi-class classifier for quality_failure_reason
               (trained only on failed batches so the model learns
               which failure mode applies).

    Uses the same time-based split as yield_predictor (first 80% / last 20%).
    NaN features are imputed with the column median inside a sklearn Pipeline.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1,
                    confusion_matrix, reason_accuracy, train_size, test_size
    """
    df = _load_data(features_path)
    train_df, test_df = _time_split(df)
    feat_cols = _feature_columns(df)

    # ---- Model A: pass/fail ------------------------------------------------
    X_train = train_df[feat_cols]
    y_train_pass = train_df[TARGET_PASS].astype(bool)
    X_test = test_df[feat_cols]
    y_test_pass = test_df[TARGET_PASS].astype(bool)

    pass_pipeline = _make_rf_pipeline(n_estimators=150)
    pass_pipeline.fit(X_train, y_train_pass)
    y_pred_pass = pass_pipeline.predict(X_test)

    accuracy = float(accuracy_score(y_test_pass, y_pred_pass))
    precision = float(precision_score(y_test_pass, y_pred_pass, zero_division=0))
    recall = float(recall_score(y_test_pass, y_pred_pass, zero_division=0))
    f1 = float(f1_score(y_test_pass, y_pred_pass, zero_division=0))
    cm = confusion_matrix(y_test_pass, y_pred_pass).tolist()
    report_pass = classification_report(y_test_pass, y_pred_pass, output_dict=True, zero_division=0)

    # ---- Model B: failure reason (failed batches only) ---------------------
    train_fail = train_df[~train_df[TARGET_PASS].astype(bool)]
    test_fail = test_df[~test_df[TARGET_PASS].astype(bool)]

    reason_metrics: dict = {}
    reason_pipeline = None

    if len(train_fail) > 0 and len(test_fail) > 0:
        X_train_fail = train_fail[feat_cols]
        y_train_reason = train_fail[TARGET_REASON].fillna("Unknown")
        X_test_fail = test_fail[feat_cols]
        y_test_reason = test_fail[TARGET_REASON].fillna("Unknown")

        reason_pipeline = _make_rf_pipeline(n_estimators=150)
        reason_pipeline.fit(X_train_fail, y_train_reason)
        y_pred_reason = reason_pipeline.predict(X_test_fail)

        reason_accuracy = float(accuracy_score(y_test_reason, y_pred_reason))
        reason_report = classification_report(
            y_test_reason, y_pred_reason, output_dict=True, zero_division=0
        )
        reason_metrics = {
            "reason_accuracy": reason_accuracy,
            "reason_classification_report": reason_report,
        }

    # ---- save artifacts ----------------------------------------------------
    out_dir = _resolve_dir(model_dir)

    # joblib.dump/load is safe: files are written to and read from a
    # controlled local directory (models/trained/) by this module only.
    joblib.dump(
        {"pipeline": pass_pipeline, "feature_columns": feat_cols},
        out_dir / PASS_MODEL_FILENAME,
    )

    if reason_pipeline is not None:
        joblib.dump(
            {
                "pipeline": reason_pipeline,
                "feature_columns": feat_cols,
                "classes": list(reason_pipeline.classes_),
            },
            out_dir / REASON_MODEL_FILENAME,
        )

    full_report = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report_pass,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        **reason_metrics,
    }
    with open(out_dir / REPORT_FILENAME, "w") as fh:
        json.dump(full_report, fh, indent=2)

    print(
        f"[quality_classifier] Accuracy={accuracy:.3f}  Precision={precision:.3f}  "
        f"Recall={recall:.3f}  F1={f1:.3f}  "
        + (
            f"Reason-Accuracy={reason_metrics.get('reason_accuracy', 0):.3f}"
            if reason_metrics
            else ""
        )
    )
    return full_report


def predict_quality_risk(
    batch_features: dict,
    model_dir: str = "models/trained",
) -> dict:
    """
    Predict pass probability and failure mode for a single batch.

    Parameters
    ----------
    batch_features : dict
        Feature values; missing keys are filled with NaN (imputed by pipeline).

    Returns
    -------
    dict:
      pass_probability   float  0-1
      risk_level         'Low' | 'Medium' | 'High'
      most_likely_failure  str or None
      failure_probabilities  {reason: probability}
    """
    out_dir = _resolve_dir(model_dir)

    # joblib.load is safe: loading only from the controlled local models/trained/ directory.
    pass_artifact = joblib.load(out_dir / PASS_MODEL_FILENAME)
    pass_pipeline: Pipeline = pass_artifact["pipeline"]
    feat_cols: list[str] = pass_artifact["feature_columns"]

    row = {col: batch_features.get(col, np.nan) for col in feat_cols}
    X = pd.DataFrame([row])

    pass_proba = pass_pipeline.predict_proba(X)[0]
    # classes_ is [False, True] after fitting on bool labels
    classes = list(pass_pipeline.named_steps["clf"].classes_)
    true_idx = classes.index(True) if True in classes else 1
    pass_probability = float(pass_proba[true_idx])

    if pass_probability > 0.85:
        risk_level = "Low"
    elif pass_probability >= 0.60:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Failure reason probabilities (only meaningful when pass_probability < 1)
    failure_probabilities: dict[str, float] = {}
    most_likely_failure: str | None = None

    reason_model_path = out_dir / REASON_MODEL_FILENAME
    if reason_model_path.exists():
        # joblib.load is safe: same controlled local directory.
        reason_artifact = joblib.load(reason_model_path)
        reason_pipeline: Pipeline = reason_artifact["pipeline"]
        reason_classes: list[str] = reason_artifact["classes"]

        reason_proba = reason_pipeline.predict_proba(X)[0]
        failure_probabilities = {
            cls: round(float(p), 4)
            for cls, p in zip(reason_classes, reason_proba)
        }
        most_likely_failure = reason_classes[int(np.argmax(reason_proba))]

    return {
        "pass_probability": round(pass_probability, 4),
        "risk_level": risk_level,
        "most_likely_failure": most_likely_failure if pass_probability < 0.85 else None,
        "failure_probabilities": failure_probabilities,
    }
