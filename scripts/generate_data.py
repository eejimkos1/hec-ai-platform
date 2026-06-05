"""
Full data generation pipeline.
Run: python scripts/generate_data.py

Steps:
1. Generate separation source data (waste_reception_log, laboratory_analysis,
   processing_batch, process_sensor_data, weather_conditions)
2. Generate fleet source data (port_waste_demand, fleet_status, voyage_history,
   offshore_platforms, maritime_weather, oil_market)
3. Save all to data/raw/*.csv
4. Print row counts, file sizes, and sample data for verification
"""

import sys
import os
import time

sys.path.insert(0, ".")

import pandas as pd


def _file_size_str(path: str) -> str:
    """Return human-readable file size."""
    size = os.path.getsize(path)
    if size >= 1_048_576:
        return f"{size / 1_048_576:.1f} MB"
    if size >= 1_024:
        return f"{size / 1_024:.1f} KB"
    return f"{size} B"


def _print_separator(char: str = "-", width: int = 70) -> None:
    print(char * width)


def generate_separation_data(output_dir: str) -> dict:
    """Generate and save all separation source tables."""
    print("\nGenerating separation data...")
    print("  Step 1/5: waste reception log (15,000 receptions)...")
    from data.generators.separation_data_generator import generate_all
    t0 = time.time()
    sep_data = generate_all()
    elapsed = time.time() - t0
    print(f"  Separation data generated in {elapsed:.1f}s")
    return dict(sep_data)


def generate_fleet_data(output_dir: str) -> dict:
    """Generate and save all fleet source tables."""
    print("\nGenerating fleet data...")
    from data.generators.fleet_data_generator import generate_all
    t0 = time.time()
    fleet_data = generate_all()
    elapsed = time.time() - t0
    print(f"  Fleet data generated in {elapsed:.1f}s")
    return fleet_data


def save_dataframes(data: dict, output_dir: str, label: str) -> list:
    """Save a dict of DataFrames to CSV and return list of info tuples."""
    saved = []
    for name, df in data.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        size_str = _file_size_str(path)
        saved.append((name, len(df), len(df.columns), size_str, path))
        print(f"  Saved {name}.csv  ({len(df):,} rows, {size_str})")
    return saved


def print_summary_table(all_files: list) -> None:
    """Print a formatted summary table of all generated files."""
    _print_separator("=")
    print("GENERATED FILES SUMMARY")
    _print_separator("=")
    header = f"{'Dataset':<35} {'Rows':>10} {'Cols':>6} {'Size':>10}"
    print(header)
    _print_separator("-")
    total_rows = 0
    for name, rows, cols, size_str, path in all_files:
        print(f"{name:<35} {rows:>10,} {cols:>6} {size_str:>10}")
        total_rows += rows
    _print_separator("-")
    print(f"{'TOTAL ROWS':<35} {total_rows:>10,}")
    _print_separator("=")


def print_sample(name: str, df: pd.DataFrame, n: int = 2) -> None:
    """Print a brief sample of a DataFrame for verification."""
    print(f"\n--- {name} (first {n} rows) ---")
    pd.set_option("display.max_columns", 6)
    pd.set_option("display.width", 120)
    sample = df.head(n)
    cols_to_show = list(sample.columns[:6])
    print(sample[cols_to_show].to_string(index=False))
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")


def main() -> None:
    pipeline_start = time.time()

    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")

    all_files = []

    # -- Separation data --
    try:
        sep_data = generate_separation_data(output_dir)
        print(f"\nSaving separation files to {output_dir}/...")
        saved = save_dataframes(sep_data, output_dir, "separation")
        all_files.extend(saved)

        # Print sample rows for a couple of key tables
        print_sample("waste_reception_log", sep_data["waste_reception_log"])
        print_sample("processing_batch",    sep_data["processing_batch"])

    except Exception as exc:
        print(f"\n[ERROR] Separation data generation failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # -- Fleet data --
    try:
        fleet_data = generate_fleet_data(output_dir)
        print(f"\nSaving fleet files to {output_dir}/...")
        saved = save_dataframes(fleet_data, output_dir, "fleet")
        all_files.extend(saved)

        # Print sample rows for a couple of key tables
        print_sample("port_waste_demand", fleet_data["port_waste_demand"])
        print_sample("voyage_history",    fleet_data["voyage_history"])

    except Exception as exc:
        print(f"\n[ERROR] Fleet data generation failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # -- Summary --
    print_summary_table(all_files)

    total_elapsed = time.time() - pipeline_start
    print(f"\nTotal pipeline runtime: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print("Done. All raw data files saved to data/raw/")


if __name__ == "__main__":
    main()
