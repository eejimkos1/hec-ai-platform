"""
Feature engineering pipeline.
Run: python scripts/engineer_features.py

Steps:
1. Check that raw data exists (if not, print error and suggest running
   generate_data.py first)
2. Build separation feature matrix -> data/features/separation_features.csv
3. Build fleet feature matrix      -> data/features/fleet_features.csv
4. Build route feature matrix      -> data/features/route_features.csv
5. Print feature counts, null percentages, and distribution summaries
"""

import sys
import os
import time

sys.path.insert(0, ".")

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_separator(char: str = "-", width: int = 70) -> None:
    print(char * width)


def _check_raw_data() -> bool:
    """Return True if all required raw data files are present."""
    raw_dir = os.path.join("data", "raw")
    required = [
        "waste_reception_log.csv",
        "laboratory_analysis.csv",
        "processing_batch.csv",
        "process_sensor_data.csv",
        "weather_conditions.csv",
        "port_waste_demand.csv",
        "fleet_status.csv",
        "voyage_history.csv",
        "offshore_platforms.csv",
        "maritime_weather.csv",
        "oil_market.csv",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(raw_dir, f))]
    if missing:
        print("\n[ERROR] Missing raw data files:")
        for f in missing:
            print(f"  - {f}")
        print("\nRun 'python scripts/generate_data.py' first to generate raw data.")
        return False
    return True


def _profile_features(df: pd.DataFrame, name: str) -> None:
    """Print null percentage, feature count, and distribution summary."""
    n_rows, n_cols = df.shape
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / n_rows * 100).round(1)

    numeric_df = df.select_dtypes(include=[np.number])
    n_numeric = len(numeric_df.columns)
    n_with_nulls = (null_counts > 0).sum()

    print(f"\n{'=' * 60}")
    print(f"FEATURE MATRIX: {name}")
    print(f"{'=' * 60}")
    print(f"  Rows:           {n_rows:,}")
    print(f"  Total columns:  {n_cols}")
    print(f"  Numeric cols:   {n_numeric}")
    print(f"  Cols with NaN:  {n_with_nulls}")

    if n_with_nulls > 0:
        top_null = null_pcts[null_pcts > 0].sort_values(ascending=False).head(5)
        print(f"  Top columns with nulls:")
        for col, pct in top_null.items():
            print(f"    {col:<45} {pct:>6.1f}% null")

    # Distribution summary for key numeric columns (first 6)
    if n_numeric > 0:
        summary_cols = list(numeric_df.columns[:6])
        print(f"\n  Distribution summary (first 6 numeric features):")
        desc = numeric_df[summary_cols].describe().loc[["mean", "std", "min", "max"]]
        for col in summary_cols:
            mean_v = desc.loc["mean", col]
            std_v  = desc.loc["std", col]
            min_v  = desc.loc["min", col]
            max_v  = desc.loc["max", col]
            print(f"    {col:<40}  mean={mean_v:>9.2f}  std={std_v:>9.2f}  "
                  f"[{min_v:.2f}, {max_v:.2f}]")


def _file_size_str(path: str) -> str:
    size = os.path.getsize(path)
    if size >= 1_048_576:
        return f"{size / 1_048_576:.1f} MB"
    if size >= 1_024:
        return f"{size / 1_024:.1f} KB"
    return f"{size} B"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    pipeline_start = time.time()

    # Check prerequisites
    if not _check_raw_data():
        sys.exit(1)

    features_dir = os.path.join("data", "features")
    os.makedirs(features_dir, exist_ok=True)

    raw_dir = "data/raw"
    ref_dir = "data/reference"
    summary_rows = []

    # ------------------------------------------------------------------ #
    # Step 2: Separation feature matrix                                   #
    # ------------------------------------------------------------------ #
    print("\nBuilding separation feature matrix...")
    sep_output = os.path.join(features_dir, "separation_features.csv")
    t0 = time.time()
    try:
        from features.separation_features import build_separation_feature_matrix
        sep_df = build_separation_feature_matrix(
            raw_data_dir=raw_dir,
            reference_dir=ref_dir,
            output_path=sep_output,
        )
        elapsed = time.time() - t0
        size_str = _file_size_str(sep_output)
        print(f"  Separation features: {len(sep_df):,} rows x {len(sep_df.columns)} cols "
              f"  [{size_str}, {elapsed:.1f}s]")
        summary_rows.append(("separation_features.csv", len(sep_df), len(sep_df.columns),
                              size_str))
        _profile_features(sep_df, "separation_features")
    except Exception as exc:
        print(f"\n[ERROR] Separation feature engineering failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 3: Fleet feature matrix                                        #
    # ------------------------------------------------------------------ #
    print("\n\nBuilding fleet feature matrix...")
    fleet_output = os.path.join(features_dir, "fleet_features.csv")
    t0 = time.time()
    try:
        from features.fleet_features import build_fleet_feature_matrix
        fleet_df = build_fleet_feature_matrix(
            raw_data_dir=raw_dir,
            reference_dir=ref_dir,
            output_path=fleet_output,
        )
        elapsed = time.time() - t0
        size_str = _file_size_str(fleet_output)
        print(f"  Fleet features: {len(fleet_df):,} rows x {len(fleet_df.columns)} cols "
              f"  [{size_str}, {elapsed:.1f}s]")
        summary_rows.append(("fleet_features.csv", len(fleet_df), len(fleet_df.columns),
                              size_str))
        _profile_features(fleet_df, "fleet_features")
    except Exception as exc:
        print(f"\n[ERROR] Fleet feature engineering failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 4: Route feature matrix                                        #
    # ------------------------------------------------------------------ #
    print("\n\nBuilding route feature matrix...")
    route_output = os.path.join(features_dir, "route_features.csv")
    t0 = time.time()
    try:
        from features.fleet_features import build_route_feature_matrix
        route_df = build_route_feature_matrix(
            raw_data_dir=raw_dir,
            reference_dir=ref_dir,
            output_path=route_output,
        )
        elapsed = time.time() - t0
        size_str = _file_size_str(route_output)
        print(f"  Route features: {len(route_df):,} rows x {len(route_df.columns)} cols "
              f"  [{size_str}, {elapsed:.1f}s]")
        summary_rows.append(("route_features.csv", len(route_df), len(route_df.columns),
                              size_str))
        _profile_features(route_df, "route_features")
    except Exception as exc:
        print(f"\n[ERROR] Route feature engineering failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 70}")
    print("FEATURE ENGINEERING SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Feature File':<35} {'Rows':>10} {'Cols':>6} {'Size':>10}")
    print(f"{'-' * 65}")
    for fname, rows, cols, size_str in summary_rows:
        print(f"{fname:<35} {rows:>10,} {cols:>6} {size_str:>10}")
    print(f"{'=' * 70}")

    total_elapsed = time.time() - pipeline_start
    print(f"\nTotal pipeline runtime: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print("Done. All feature files saved to data/features/")


if __name__ == "__main__":
    main()
