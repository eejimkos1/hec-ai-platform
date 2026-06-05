"""Run the full data pipeline (generate → features → models).

Called at app startup if data/models are missing.
Can also be run standalone: python run_pipeline.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def pipeline_needed() -> bool:
    """Check if any pipeline output is missing."""
    checks = [
        ROOT / "data" / "raw" / "waste_reception_log.csv",
        ROOT / "data" / "features" / "separation_features.csv",
        ROOT / "models" / "trained" / "yield_model.joblib",
    ]
    return not all(p.exists() for p in checks)


def run_pipeline():
    """Execute all three pipeline stages."""
    scripts = [
        ROOT / "scripts" / "generate_data.py",
        ROOT / "scripts" / "engineer_features.py",
        ROOT / "scripts" / "train_models.py",
    ]
    for script in scripts:
        print(f"Running {script.name}...")
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"ERROR in {script.name}:\n{result.stderr}")
            return False
        print(f"  ✓ {script.name} complete")
    return True


if __name__ == "__main__":
    if pipeline_needed():
        print("Pipeline outputs missing — running full pipeline...")
        success = run_pipeline()
        if success:
            print("\n✓ Pipeline complete. Run: streamlit run app/main.py")
        else:
            print("\n✗ Pipeline failed.")
            sys.exit(1)
    else:
        print("All pipeline outputs present. Nothing to do.")
