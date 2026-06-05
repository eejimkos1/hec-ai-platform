"""Application configuration — paths, constants, column descriptions, display settings."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DATA_REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"

# Display
PAGE_TITLE = "HEC AI Platform"
PAGE_ICON = "🛢️"
LAYOUT = "wide"

# Color scheme (for charts)
COLORS = {
    "primary": "#0D1B2A",
    "primary_light": "#1B2838",
    "accent": "#1B9AAA",
    "accent_light": "#23C4D8",
    "success": "#2E7D32",
    "danger": "#C62828",
    "warning": "#F9A825",
    "info": "#1B9AAA",
    "text_dark": "#0D1B2A",
    "text_muted": "#546E7A",
    "bg_white": "#FFFFFF",
    "bg_light": "#F5F7FA",
    "border": "#E0E4E8",
    "facilities": {
        "PIR": "#0D1B2A",
        "HAM": "#1B9AAA",
        "GIB": "#2E7D32",
        "MLT": "#5C6BC0",
    },
}

PLOTLY_LAYOUT = dict(
    font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12),
    colorway=["#0D1B2A", "#1B9AAA", "#2E7D32", "#5C6BC0", "#F9A825", "#C62828", "#78909C", "#4DB6AC"],
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor="#0D1B2A", font_color="white", font_size=12),
    margin=dict(t=40, b=40, l=50, r=20),
)

# Financial constants
EUR_PER_YIELD_PCT = 1_500.0
AVG_BATCHES_PER_YEAR = 12_000
ACCEPTANCE_FEE_EUR_M3 = 30.0
RECOVERY_FRACTION = 0.30
RECOVERY_PRICE_EUR_M3 = 400.0
PORT_CHARGES_EUR = 1_500.0
FUEL_PRICE_FALLBACK_USD_MT = 500.0

# Pipeline step labels
PIPELINE_STEPS = [
    {"label": "1. Source Data", "icon": "📊", "description": "Raw data from operational systems"},
    {"label": "2. Feature Engineering", "icon": "⚙️", "description": "Computed features for ML"},
    {"label": "3. ML Model", "icon": "🤖", "description": "Trained model performance"},
    {"label": "4. Results", "icon": "🎯", "description": "Predictions and optimizations"},
]

PIPELINE_STEPS_5 = [
    {"label": "1. Source Data", "icon": "📊", "description": "Raw operational data"},
    {"label": "2. Analytics", "icon": "📈", "description": "Descriptive insights"},
    {"label": "3. Feature Engineering", "icon": "⚙️", "description": "Computed ML features"},
    {"label": "4. ML Model", "icon": "🤖", "description": "Training & performance"},
    {"label": "5. Results", "icon": "🎯", "description": "Predictions & optimization"},
]

# Column descriptions — used for tooltips in data tables
# Format: {column_name: "Business explanation (Source System)"}
SEPARATION_COLUMN_DESCRIPTIONS = {
    "reception_id": "Unique waste collection event ID (Waste Acceptance System)",
    "facility_code": "Treatment facility: PIR=Piraeus, HAM=Hamburg, GIB=Gibraltar, MLT=Malta",
    "waste_subcategory": "MARPOL waste classification type",
    "actual_volume_m3": "Measured volume of waste received in cubic meters",
    "viscosity_40c_cst": "Kinematic viscosity at 40°C in centistokes (LIMS - ASTM D445)",
    "water_content_pct": "Water fraction percentage (LIMS - Karl Fischer titration)",
    "oil_content_pct": "Oil/hydrocarbon fraction percentage (LIMS - extraction method)",
    "solids_content_pct": "Solid sediment percentage (LIMS - filtration at 105°C)",
    "density_15c_kg_m3": "Density at standard 15°C in kg/m³ (LIMS - ASTM D4052)",
    "flash_point_c": "Flash point in °C — safety critical (LIMS - Pensky-Martens ASTM D93)",
    "sulfur_total_pct": "Total sulfur content (LIMS - XRF, ASTM D4294)",
    "oil_recovery_yield_pct": "PRIMARY TARGET: Percentage of oil recovered vs. oil in feed",
    "quality_pass": "Did the output meet buyer specification? (ERP)",
    "net_margin_eur": "Revenue minus processing cost in EUR (ERP)",
    "viscosity_temperature_ratio": "FEATURE: viscosity/temperature — normalized pumpability",
    "emulsion_difficulty_score": "FEATURE: Composite difficulty of emulsion breaking",
    "contamination_index": "FEATURE: (V+Ni+Fe+Na)/1000 — metal contamination severity",
    "specific_energy_input": "FEATURE: kWh per m³ processed — energy intensity",
    "capacity_utilization": "FEATURE: actual flow / max capacity — equipment loading",
    "rolling_yield_7d": "FEATURE: 7-day rolling average yield at this facility",
}

FLEET_COLUMN_DESCRIPTIONS = {
    "port_code": "Port identifier (AIS/Port Management System)",
    "vessel_calls_total": "Total vessel arrivals at port that day (AIS)",
    "waste_volume_collected_m3": "TARGET: Total waste collected at port that day (m³)",
    "current_storage_fill_pct": "Port waste storage fill level percentage",
    "days_until_full": "Days until port storage overflows at current fill rate",
    "vessel_calls_7d_rolling": "FEATURE: 7-day rolling average of vessel calls",
    "waste_7d_rolling": "FEATURE: 7-day rolling average of waste collected",
    "cruise_season_flag": "FEATURE: 1 if Apr-Oct at Mediterranean port",
    "urgency_score": "FEATURE: fill_pct × (1/days_until_full) — collection urgency",
    "brent_crude_usd_bbl": "Brent crude oil price USD/barrel (Market Data - Platts)",
    "shipping_demand_index": "Shipping activity index (Market Data - Baltic proxy)",
}


def check_data_status() -> dict:
    """Check which data files exist. Returns dict of {category: {file: exists_bool}}."""
    status = {"raw": {}, "features": {}, "models": {}}

    raw_files = [
        "waste_reception_log.csv",
        "laboratory_analysis.csv",
        "processing_batch.csv",
        "port_waste_demand.csv",
        "voyage_history.csv",
        "oil_market.csv",
    ]
    for f in raw_files:
        status["raw"][f] = (DATA_RAW_DIR / f).exists()

    feature_files = ["separation_features.csv", "fleet_features.csv", "route_features.csv"]
    for f in feature_files:
        status["features"][f] = (DATA_FEATURES_DIR / f).exists()

    model_files = [
        "yield_model.joblib",
        "quality_classifier.joblib",
        "demand_forecaster.joblib",
        "profitability_model.joblib",
    ]
    for f in model_files:
        status["models"][f] = (MODELS_DIR / f).exists()

    return status
