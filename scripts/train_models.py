"""
Model training pipeline.
Run: python scripts/train_models.py

Steps:
1. Check that feature data exists
2. Train yield predictor      -> print R², RMSE, MAE
3. Train quality classifier   -> print accuracy, F1
4. Train demand forecaster    -> print MAPE, R²
5. Train profitability model  -> print R², RMSE
6. Build route optimizer
7. Print summary table of all model metrics
"""

import sys
import os
import time

sys.path.insert(0, ".")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_separator(char: str = "-", width: int = 70) -> None:
    print(char * width)


def _check_features() -> bool:
    """Return True if all required feature files are present."""
    features_dir = os.path.join("data", "features")
    required = [
        "separation_features.csv",
        "fleet_features.csv",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(features_dir, f))]
    if missing:
        print("\n[ERROR] Missing feature files:")
        for f in missing:
            print(f"  - {f}")
        print("\nRun 'python scripts/engineer_features.py' first to build features.")
        return False

    # Also need voyage_history for profitability model
    voyage_path = os.path.join("data", "raw", "voyage_history.csv")
    if not os.path.exists(voyage_path):
        print(f"\n[ERROR] Missing raw data file: {voyage_path}")
        print("Run 'python scripts/generate_data.py' first to generate raw data.")
        return False

    return True


def _fmt_metric(value: float, fmt: str = ".4f") -> str:
    return f"{value:{fmt}}" if value is not None else "N/A"


# ---------------------------------------------------------------------------
# Individual training steps
# ---------------------------------------------------------------------------

def train_yield_predictor(model_dir: str) -> dict:
    print("\n[1/5] Training Yield Predictor (XGBoost regressor)...")
    t0 = time.time()
    from models.separation.yield_predictor import train_and_save
    metrics = train_and_save(
        features_path="data/features/separation_features.csv",
        model_dir=model_dir,
    )
    elapsed = time.time() - t0
    print(f"  R²   = {_fmt_metric(metrics.get('r2'))}")
    print(f"  RMSE = {_fmt_metric(metrics.get('rmse'))}")
    print(f"  MAE  = {_fmt_metric(metrics.get('mae'))}")
    print(f"  Train rows: {metrics.get('train_size', '?'):,}  |  "
          f"Test rows: {metrics.get('test_size', '?'):,}")
    print(f"  Elapsed: {elapsed:.1f}s")
    return metrics


def train_quality_classifier(model_dir: str) -> dict:
    print("\n[2/5] Training Quality Classifier (Random Forest)...")
    t0 = time.time()
    from models.separation.quality_classifier import train_and_save
    metrics = train_and_save(
        features_path="data/features/separation_features.csv",
        model_dir=model_dir,
    )
    elapsed = time.time() - t0
    print(f"  Accuracy  = {_fmt_metric(metrics.get('accuracy'))}")
    print(f"  F1        = {_fmt_metric(metrics.get('f1'))}")
    print(f"  Precision = {_fmt_metric(metrics.get('precision'))}")
    print(f"  Recall    = {_fmt_metric(metrics.get('recall'))}")
    print(f"  Train rows: {metrics.get('train_size', '?'):,}  |  "
          f"Test rows: {metrics.get('test_size', '?'):,}")
    print(f"  Elapsed: {elapsed:.1f}s")
    return metrics


def train_demand_forecaster(model_dir: str) -> dict:
    print("\n[3/5] Training Demand Forecaster (XGBoost regressor)...")
    t0 = time.time()
    from models.fleet.demand_forecaster import train_and_save
    metrics = train_and_save(
        features_path="data/features/fleet_features.csv",
        model_dir=model_dir,
    )
    elapsed = time.time() - t0
    print(f"  MAPE = {_fmt_metric(metrics.get('mape'))}")
    print(f"  RMSE = {_fmt_metric(metrics.get('rmse'))}")
    print(f"  R²   = {_fmt_metric(metrics.get('r2'))}")
    print(f"  Elapsed: {elapsed:.1f}s")
    return metrics


def train_profitability_model(model_dir: str) -> dict:
    print("\n[4/5] Training Profitability Model (XGBoost regressor)...")
    t0 = time.time()
    from models.fleet.profitability_model import train_and_save
    metrics = train_and_save(
        voyage_path="data/raw/voyage_history.csv",
        model_dir=model_dir,
    )
    elapsed = time.time() - t0
    print(f"  R²   = {_fmt_metric(metrics.get('r2'))}")
    print(f"  RMSE = {_fmt_metric(metrics.get('rmse'))}")
    print(f"  MAE  = {_fmt_metric(metrics.get('mae'))}")
    print(f"  Train rows: {metrics.get('train_rows', '?'):,}  |  "
          f"Test rows: {metrics.get('test_rows', '?'):,}")
    print(f"  Elapsed: {elapsed:.1f}s")
    return metrics


def build_route_optimizer(model_dir: str) -> dict:
    print("\n[5/5] Building Route Optimizer (config precomputation)...")
    t0 = time.time()
    from models.fleet.route_optimizer import build_optimizer
    result = build_optimizer(model_dir=model_dir)
    elapsed = time.time() - t0
    print(f"  Distance pairs: {result.get('distance_lookup_size', '?'):,}")
    print(f"  Fleet vessels:  {result.get('fleet_count', '?')}")
    print(f"  Port coords:    {result.get('port_count', '?')}")
    print(f"  Elapsed: {elapsed:.1f}s")
    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_model_summary(results: dict) -> None:
    _print_separator("=")
    print("MODEL TRAINING SUMMARY")
    _print_separator("=")

    # Yield predictor
    m = results.get("yield_predictor", {})
    print(f"{'Yield Predictor':<30}  "
          f"R²={_fmt_metric(m.get('r2'), '.4f')}  "
          f"RMSE={_fmt_metric(m.get('rmse'), '.3f')}  "
          f"MAE={_fmt_metric(m.get('mae'), '.3f')}")

    # Quality classifier
    m = results.get("quality_classifier", {})
    print(f"{'Quality Classifier':<30}  "
          f"Accuracy={_fmt_metric(m.get('accuracy'), '.4f')}  "
          f"F1={_fmt_metric(m.get('f1'), '.4f')}")

    # Demand forecaster
    m = results.get("demand_forecaster", {})
    print(f"{'Demand Forecaster':<30}  "
          f"R²={_fmt_metric(m.get('r2'), '.4f')}  "
          f"RMSE={_fmt_metric(m.get('rmse'), '.3f')}  "
          f"MAPE={_fmt_metric(m.get('mape'), '.4f')}")

    # Profitability model
    m = results.get("profitability_model", {})
    print(f"{'Profitability Model':<30}  "
          f"R²={_fmt_metric(m.get('r2'), '.4f')}  "
          f"RMSE={_fmt_metric(m.get('rmse'), '.3f')}")

    # Route optimizer
    m = results.get("route_optimizer", {})
    print(f"{'Route Optimizer':<30}  "
          f"dist_pairs={m.get('distance_lookup_size', 'N/A')}  "
          f"vessels={m.get('fleet_count', 'N/A')}  "
          f"ports={m.get('port_count', 'N/A')}")

    _print_separator("=")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pipeline_start = time.time()

    # Check prerequisites
    if not _check_features():
        sys.exit(1)

    model_dir = "models/trained"
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model output directory: {os.path.abspath(model_dir)}")

    results = {}
    errors = []

    # Train yield predictor
    try:
        results["yield_predictor"] = train_yield_predictor(model_dir)
    except Exception as exc:
        print(f"\n[ERROR] Yield predictor training failed: {exc}")
        import traceback
        traceback.print_exc()
        errors.append("yield_predictor")
        results["yield_predictor"] = {}

    # Train quality classifier
    try:
        results["quality_classifier"] = train_quality_classifier(model_dir)
    except Exception as exc:
        print(f"\n[ERROR] Quality classifier training failed: {exc}")
        import traceback
        traceback.print_exc()
        errors.append("quality_classifier")
        results["quality_classifier"] = {}

    # Train demand forecaster
    try:
        results["demand_forecaster"] = train_demand_forecaster(model_dir)
    except Exception as exc:
        print(f"\n[ERROR] Demand forecaster training failed: {exc}")
        import traceback
        traceback.print_exc()
        errors.append("demand_forecaster")
        results["demand_forecaster"] = {}

    # Train profitability model
    try:
        results["profitability_model"] = train_profitability_model(model_dir)
    except Exception as exc:
        print(f"\n[ERROR] Profitability model training failed: {exc}")
        import traceback
        traceback.print_exc()
        errors.append("profitability_model")
        results["profitability_model"] = {}

    # Build route optimizer
    try:
        results["route_optimizer"] = build_route_optimizer(model_dir)
    except Exception as exc:
        print(f"\n[ERROR] Route optimizer build failed: {exc}")
        import traceback
        traceback.print_exc()
        errors.append("route_optimizer")
        results["route_optimizer"] = {}

    # Summary
    print_model_summary(results)

    total_elapsed = time.time() - pipeline_start
    print(f"\nTotal pipeline runtime: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    if errors:
        print(f"\n[WARNING] {len(errors)} model(s) failed to train: {', '.join(errors)}")
        sys.exit(1)
    else:
        print("Done. All models trained and saved to models/trained/")


if __name__ == "__main__":
    main()
