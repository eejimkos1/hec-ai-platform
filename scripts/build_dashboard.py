"""Build static dashboard data — aggregates CSVs and model metrics into compact JSON files."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)

DATA_RAW = ROOT / "data" / "raw"
DATA_FEAT = ROOT / "data" / "features"
MODELS = ROOT / "models" / "trained"


def build_separation_json():
    """Aggregate separation features + model outputs into compact JSON."""
    sep = pd.read_csv(DATA_FEAT / "separation_features.csv", parse_dates=["processing_date"])

    # Yield histogram (50 bins)
    counts, edges = np.histogram(sep["oil_recovery_yield_pct"].dropna(), bins=50)
    yield_hist = {"counts": counts.tolist(), "edges": edges.tolist()}

    # Feature importances (top 15)
    fi = pd.read_csv(MODELS / "yield_feature_importances.csv")
    fi = fi.sort_values("importance", ascending=False).head(15)
    feat_imp = {"features": fi["feature"].tolist(), "importances": fi["importance"].tolist()}

    # Predictions scatter (sample 500)
    preds = pd.read_csv(MODELS / "yield_predictions_vs_actuals.csv")
    sample = preds.sample(min(500, len(preds)), random_state=42)
    pred_scatter = {"actual": sample["actual"].round(2).tolist(), "predicted": sample["predicted"].round(2).tolist()}

    # Facility stats
    fac_stats = (
        sep.groupby("facility_code")
        .agg(
            avg_yield=("oil_recovery_yield_pct", "mean"),
            avg_margin=("net_margin_eur", "mean"),
            total_batches=("oil_recovery_yield_pct", "count"),
            quality_rate=("quality_pass", "mean"),
        )
        .round(2)
        .reset_index()
    )
    facility_data = fac_stats.to_dict(orient="records")

    # Monthly margins
    monthly = (
        sep.set_index("processing_date")
        .resample("ME")
        .agg(avg_yield=("oil_recovery_yield_pct", "mean"), total_margin=("net_margin_eur", "sum"), batches=("net_margin_eur", "count"))
        .round(2)
        .reset_index()
    )
    monthly["processing_date"] = monthly["processing_date"].dt.strftime("%Y-%m-%d")
    monthly_data = monthly.to_dict(orient="records")

    # Quality breakdown
    quality_pass = int(sep["quality_pass"].sum())
    quality_fail = int(len(sep) - quality_pass)

    # Yield by facility (box plot data)
    yield_by_fac = {}
    for fac, grp in sep.groupby("facility_code"):
        yield_by_fac[fac] = grp["oil_recovery_yield_pct"].dropna().round(2).tolist()

    result = {
        "yield_histogram": yield_hist,
        "feature_importances": feat_imp,
        "predictions_scatter": pred_scatter,
        "facility_stats": facility_data,
        "monthly_trends": monthly_data,
        "quality_breakdown": {"pass": quality_pass, "fail": quality_fail},
        "yield_by_facility": yield_by_fac,
        "total_batches": len(sep),
        "avg_yield": round(sep["oil_recovery_yield_pct"].mean(), 2),
        "avg_margin": round(sep["net_margin_eur"].mean(), 2),
    }

    with open(OUT / "separation.json", "w") as f:
        json.dump(result, f)
    print(f"  separation.json ({(OUT / 'separation.json').stat().st_size / 1024:.0f} KB)")


def build_fleet_json():
    """Aggregate fleet features + voyage data into compact JSON."""
    fleet = pd.read_csv(DATA_FEAT / "fleet_features.csv", parse_dates=["date"])
    voyages = pd.read_csv(DATA_RAW / "voyage_history.csv", parse_dates=["departure_time"])

    # Port-level demand averages
    port_stats = (
        fleet.groupby("port_code")
        .agg(
            avg_demand=("waste_volume_collected_m3", "mean"),
            total_demand=("waste_volume_collected_m3", "sum"),
            days=("date", "count"),
        )
        .round(1)
        .reset_index()
    )
    port_stats_data = port_stats.to_dict(orient="records")

    # Voyage economics by vessel class
    if "vessel_class" in voyages.columns:
        voy_class = (
            voyages.groupby("vessel_class")
            .agg(
                count=("voyage_id", "count"),
                avg_revenue=("revenue_eur", "mean"),
                avg_margin=("voyage_margin_eur", "mean"),
                avg_fuel=("fuel_cost_eur", "mean"),
            )
            .round(0)
            .reset_index()
        )
        voyage_by_class = voy_class.to_dict(orient="records")
    else:
        voyage_by_class = []

    # Monthly demand trends (top 5 ports)
    top_ports = port_stats.nlargest(5, "total_demand")["port_code"].tolist()
    monthly_demand = {}
    for port in top_ports:
        port_df = fleet[fleet["port_code"] == port].set_index("date").resample("ME")["waste_volume_collected_m3"].sum().reset_index()
        port_df["date"] = port_df["date"].dt.strftime("%Y-%m-%d")
        monthly_demand[port] = port_df.to_dict(orient="records")

    # Demand predictions scatter (sample 500)
    pred_path = MODELS / "demand_forecaster_predictions.csv"
    if pred_path.exists():
        dpreds = pd.read_csv(pred_path)
        dsample = dpreds.sample(min(500, len(dpreds)), random_state=42)
        demand_scatter = {"actual": dsample["waste_volume_collected_m3"].round(1).tolist(), "predicted": dsample["predicted_volume_m3"].round(1).tolist()}
    else:
        demand_scatter = {"actual": [], "predicted": []}

    # Port metrics from forecaster
    port_metrics_path = MODELS / "demand_forecaster_port_metrics.csv"
    if port_metrics_path.exists():
        port_metrics = pd.read_csv(port_metrics_path).to_dict(orient="records")
    else:
        port_metrics = []

    result = {
        "port_stats": port_stats_data,
        "voyage_by_class": voyage_by_class,
        "monthly_demand_top5": monthly_demand,
        "demand_scatter": demand_scatter,
        "port_metrics": port_metrics,
        "total_voyages": len(voyages),
        "total_revenue": round(voyages["revenue_eur"].sum(), 0) if "revenue_eur" in voyages.columns else 0,
        "avg_voyage_margin": round(voyages["voyage_margin_eur"].mean(), 0) if "voyage_margin_eur" in voyages.columns else 0,
        "total_fuel_cost": round(voyages["fuel_cost_eur"].sum(), 0) if "fuel_cost_eur" in voyages.columns else 0,
    }

    with open(OUT / "fleet.json", "w") as f:
        json.dump(result, f)
    print(f"  fleet.json ({(OUT / 'fleet.json').stat().st_size / 1024:.0f} KB)")


def build_models_json():
    """Combine all model metric files."""
    result = {}

    for name, fname in [
        ("yield", "yield_evaluation_metrics.json"),
        ("demand", "demand_forecaster_metrics.json"),
        ("profitability", "profitability_model_metrics.json"),
        ("quality", "classification_report.json"),
    ]:
        p = MODELS / fname
        if p.exists():
            with open(p) as f:
                result[name] = json.load(f)
        else:
            result[name] = {}

    with open(OUT / "models.json", "w") as f:
        json.dump(result, f)
    print(f"  models.json ({(OUT / 'models.json').stat().st_size / 1024:.0f} KB)")


def build_financial_json():
    """Combined financial overview KPIs."""
    sep = pd.read_csv(DATA_FEAT / "separation_features.csv")
    voyages = pd.read_csv(DATA_RAW / "voyage_history.csv", parse_dates=["departure_time"])

    # Separation economics
    total_batches = len(sep)
    avg_yield = round(sep["oil_recovery_yield_pct"].mean(), 2)
    avg_margin = round(sep["net_margin_eur"].mean(), 0)
    quality_rate = round(sep["quality_pass"].mean() * 100, 1)

    # Fleet economics
    total_voyages = len(voyages)
    total_revenue = round(voyages["revenue_eur"].sum(), 0) if "revenue_eur" in voyages.columns else 0
    total_fuel = round(voyages["fuel_cost_eur"].sum(), 0) if "fuel_cost_eur" in voyages.columns else 0
    avg_voy_margin = round(voyages["voyage_margin_eur"].mean(), 0) if "voyage_margin_eur" in voyages.columns else 0

    # AI opportunity
    yield_improvement_pp = 1.07
    eur_per_yield_pct = 1500
    avg_batches_year = 12000
    annual_sep_savings = round(yield_improvement_pp * eur_per_yield_pct * avg_batches_year, 0)
    annual_fleet_savings = round(total_fuel / 4.5 * 0.10, 0) if total_fuel > 0 else 0
    total_opportunity = annual_sep_savings + annual_fleet_savings

    # Monthly combined margins
    sep_full = pd.read_csv(DATA_FEAT / "separation_features.csv", parse_dates=["processing_date"])
    sep_monthly = (
        sep_full.set_index("processing_date").resample("ME")["net_margin_eur"].sum().reset_index()
    )
    sep_monthly.columns = ["date", "separation_margin"]

    fleet_monthly = (
        voyages.set_index("departure_time").resample("ME")["voyage_margin_eur"].sum().reset_index()
    )
    fleet_monthly.columns = ["date", "fleet_margin"]

    sep_monthly["date"] = sep_monthly["date"].dt.strftime("%Y-%m-%d")
    fleet_monthly["date"] = fleet_monthly["date"].dt.strftime("%Y-%m-%d")

    result = {
        "separation": {
            "total_batches": total_batches,
            "avg_yield": avg_yield,
            "avg_margin": avg_margin,
            "quality_rate": quality_rate,
        },
        "fleet": {
            "total_voyages": total_voyages,
            "total_revenue": total_revenue,
            "total_fuel_cost": total_fuel,
            "avg_voyage_margin": avg_voy_margin,
        },
        "ai_opportunity": {
            "total": total_opportunity,
            "separation_savings": annual_sep_savings,
            "fleet_savings": annual_fleet_savings,
        },
        "monthly_separation": sep_monthly.to_dict(orient="records"),
        "monthly_fleet": fleet_monthly.to_dict(orient="records"),
    }

    with open(OUT / "financial.json", "w") as f:
        json.dump(result, f)
    print(f"  financial.json ({(OUT / 'financial.json').stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    print("Building dashboard data...")
    build_separation_json()
    build_fleet_json()
    build_models_json()
    build_financial_json()
    print("Done!")
