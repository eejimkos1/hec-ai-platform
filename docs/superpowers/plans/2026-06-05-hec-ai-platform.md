# HEC AI Platform — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete Streamlit application with realistic synthetic data, feature engineering pipelines, ML models, and dashboard views for HEC's separation process control and fleet routing optimization.

**Architecture:** Python monorepo with data generators producing CSVs, feature engineering modules transforming them, ML models trained on the enriched data, and a Streamlit multi-page app displaying the full pipeline (Source → Features → Model → Result) for each use case.

**Tech Stack:** Python 3.11+, Streamlit, pandas, numpy, scikit-learn, XGBoost, Prophet, scipy, plotly, folium/pydeck (maps)

---

## File Structure

```
C:\SAPDevelop\meli\
├── requirements.txt
├── app/
│   ├── main.py
│   ├── pages/
│   │   ├── 1_separation_control.py
│   │   └── 2_fleet_routing.py
│   ├── components/
│   │   ├── pipeline_viewer.py
│   │   ├── map_viewer.py
│   │   └── charts.py
│   └── config.py
├── data/
│   ├── generators/
│   │   ├── common.py
│   │   ├── separation_data_generator.py
│   │   └── fleet_data_generator.py
│   ├── raw/                  (generated CSVs)
│   ├── features/             (feature-engineered CSVs)
│   └── reference/            (static lookup tables)
├── models/
│   ├── separation/
│   │   ├── yield_predictor.py
│   │   ├── parameter_optimizer.py
│   │   └── quality_classifier.py
│   ├── fleet/
│   │   ├── demand_forecaster.py
│   │   ├── route_optimizer.py
│   │   └── profitability_model.py
│   └── trained/              (serialized .joblib files)
├── features/
│   ├── separation_features.py
│   └── fleet_features.py
└── scripts/
    ├── generate_data.py
    ├── engineer_features.py
    └── train_models.py
```

---

## Task 1: Project Setup & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `app/__init__.py`, `data/__init__.py`, `data/generators/__init__.py`, `models/__init__.py`, `models/separation/__init__.py`, `models/fleet/__init__.py`, `features/__init__.py`, `scripts/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
prophet>=1.1.4
scipy>=1.11.0
plotly>=5.17.0
pydeck>=0.8.0
folium>=0.14.0
streamlit-folium>=0.15.0
joblib>=1.3.0
openpyxl>=3.1.0
```

- [ ] **Step 2: Create directory structure and __init__.py files**

```bash
mkdir -p app/pages app/components data/generators data/raw data/features data/reference models/separation models/fleet models/trained features scripts
touch app/__init__.py data/__init__.py data/generators/__init__.py models/__init__.py models/separation/__init__.py models/fleet/__init__.py features/__init__.py scripts/__init__.py
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 4: Verify installation**

```bash
python -c "import streamlit, pandas, numpy, sklearn, xgboost, prophet, scipy, plotly, pydeck, folium, joblib; print('All dependencies OK')"
```

---

## Task 2: Reference Data & Common Utilities

**Files:**
- Create: `data/generators/common.py`
- Create: `data/reference/ports.csv`
- Create: `data/reference/equipment_master.csv`
- Create: `data/reference/port_distance_matrix.csv`
- Create: `data/reference/fleet_master.csv`
- Create: `data/reference/offshore_platforms.csv`

- [ ] **Step 1: Create common.py with shared constants and utilities**

```python
"""Shared constants, lookup tables, and utility functions for data generation."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
DATE_START = datetime(2022, 1, 1)
DATE_END = datetime(2026, 6, 1)

FACILITIES = {
    "PIR": {"name": "Piraeus", "country": "GR", "lat": 37.9475, "lon": 23.6370, "annual_volume_m3": 150000},
    "HAM": {"name": "Hamburg", "country": "DE", "lat": 53.5461, "lon": 9.9660, "annual_volume_m3": 80000},
    "GIB": {"name": "Gibraltar", "country": "GI", "lat": 36.1408, "lon": -5.3536, "annual_volume_m3": 60000},
    "MLT": {"name": "Malta (Valletta)", "country": "MT", "lat": 35.8989, "lon": 14.5146, "annual_volume_m3": 40000},
}

PORTS = {
    "PIR": {"name": "Piraeus", "country": "GR", "lat": 37.9475, "lon": 23.6370},
    "HAM": {"name": "Hamburg", "country": "DE", "lat": 53.5461, "lon": 9.9660},
    "GIB": {"name": "Gibraltar", "country": "GI", "lat": 36.1408, "lon": -5.3536},
    "MLT": {"name": "Valletta", "country": "MT", "lat": 35.8989, "lon": 14.5146},
    "GEN": {"name": "Genoa", "country": "IT", "lat": 44.4056, "lon": 8.9463},
    "MRS": {"name": "Marseille", "country": "FR", "lat": 43.3462, "lon": 5.3200},
    "BCN": {"name": "Barcelona", "country": "ES", "lat": 41.3590, "lon": 2.1685},
    "RTM": {"name": "Rotterdam", "country": "NL", "lat": 51.9054, "lon": 4.4661},
    "ALG": {"name": "Algeciras", "country": "ES", "lat": 36.1272, "lon": -5.4427},
    "LIM": {"name": "Limassol", "country": "CY", "lat": 34.6741, "lon": 33.0379},
    "ALE": {"name": "Alexandria", "country": "EG", "lat": 31.1842, "lon": 29.8763},
    "IST": {"name": "Istanbul", "country": "TR", "lat": 41.0053, "lon": 28.9770},
    "PSD": {"name": "Port Said", "country": "EG", "lat": 31.2565, "lon": 32.2841},
    "TAN": {"name": "Tangier", "country": "MA", "lat": 35.7850, "lon": -5.8029},
}

WASTE_CATEGORIES = {
    "Bilge Water": {"oil_range": (0.03, 0.15), "water_range": (0.80, 0.95), "solids_range": (0.01, 0.05)},
    "Fuel Oil Sludge": {"oil_range": (0.20, 0.60), "water_range": (0.10, 0.40), "solids_range": (0.10, 0.30)},
    "Oily Tank Washings": {"oil_range": (0.30, 0.70), "water_range": (0.20, 0.60), "solids_range": (0.02, 0.10)},
    "Dirty Ballast Water": {"oil_range": (0.001, 0.03), "water_range": (0.95, 0.99), "solids_range": (0.001, 0.01)},
    "Scale & Sludge from Tank Cleaning": {"oil_range": (0.10, 0.30), "water_range": (0.10, 0.30), "solids_range": (0.40, 0.70)},
    "Cargo Residues (Crude)": {"oil_range": (0.70, 0.95), "water_range": (0.03, 0.20), "solids_range": (0.02, 0.10)},
    "Cargo Residues (Product)": {"oil_range": (0.80, 0.98), "water_range": (0.01, 0.10), "solids_range": (0.005, 0.03)},
    "Offshore Production Slops": {"oil_range": (0.40, 0.70), "water_range": (0.20, 0.50), "solids_range": (0.05, 0.20)},
    "Engine Room Oily Water": {"oil_range": (0.02, 0.10), "water_range": (0.85, 0.97), "solids_range": (0.01, 0.05)},
    "Vegetable Oil Residues": {"oil_range": (0.60, 0.90), "water_range": (0.05, 0.30), "solids_range": (0.02, 0.10)},
    "Chemical Washing Residues": {"oil_range": (0.10, 0.50), "water_range": (0.30, 0.70), "solids_range": (0.05, 0.20)},
}

VESSEL_TYPES = {
    "Container Ship": {"gt_range": (10000, 200000), "engine_kw_range": (20000, 80000), "waste_factor": 1.0},
    "Crude Oil Tanker": {"gt_range": (30000, 160000), "engine_kw_range": (15000, 40000), "waste_factor": 2.5},
    "Product Tanker": {"gt_range": (5000, 50000), "engine_kw_range": (5000, 15000), "waste_factor": 1.8},
    "Bulk Carrier": {"gt_range": (20000, 100000), "engine_kw_range": (8000, 25000), "waste_factor": 0.8},
    "Cruise Ship": {"gt_range": (70000, 230000), "engine_kw_range": (40000, 80000), "waste_factor": 1.5},
    "LNG Carrier": {"gt_range": (80000, 170000), "engine_kw_range": (30000, 50000), "waste_factor": 0.6},
    "FPSO": {"gt_range": (100000, 200000), "engine_kw_range": (20000, 40000), "waste_factor": 5.0},
    "Offshore Platform": {"gt_range": (5000, 50000), "engine_kw_range": (5000, 20000), "waste_factor": 4.0},
    "Chemical Tanker": {"gt_range": (5000, 40000), "engine_kw_range": (5000, 15000), "waste_factor": 1.2},
    "RoRo": {"gt_range": (10000, 60000), "engine_kw_range": (10000, 30000), "waste_factor": 0.7},
}

FUEL_TYPES = ["HFO", "VLSFO", "LSMGO", "LNG"]
FUEL_TYPE_WEIGHTS = [0.25, 0.40, 0.25, 0.10]  # Post-IMO2020: VLSFO dominant

COLLECTION_METHODS = ["Barge Transfer", "Direct Pumping", "Shore Pipeline", "Truck Delivery"]
COLLECTION_METHOD_WEIGHTS = [0.35, 0.40, 0.15, 0.10]

MIXING_LEVELS = ["None", "Mild", "Moderate", "Severe"]
MIXING_WEIGHTS = [0.15, 0.35, 0.35, 0.15]

CONTRACT_TYPES = ["Spot", "Annual Contract", "Framework Agreement"]
CONTRACT_WEIGHTS = [0.40, 0.35, 0.25]

PRE_TREATMENTS = ["None", "Heating Only", "Chemical + Heating", "Gravity Pre-Settling", "Filtration"]
PRE_TREATMENT_WEIGHTS = [0.10, 0.30, 0.35, 0.20, 0.05]

TARGET_SPECS = ["Refinery Grade", "Marine Fuel Blend", "Industrial Burner", "Asphalt Blend"]
TARGET_SPEC_WEIGHTS = [0.35, 0.30, 0.25, 0.10]

BATCH_PRIORITIES = ["Normal", "Urgent", "Premium"]
BATCH_PRIORITY_WEIGHTS = [0.65, 0.25, 0.10]

SEPARATOR_TYPES = ["3-Phase Disc Stack Centrifuge", "Decanter Centrifuge", "Gravity Separator", "Plate Separator", "DAF Unit"]

HEC_VESSEL_NAMES = [
    "HEC Poseidon", "HEC Athena", "HEC Apollo", "HEC Hermes", "HEC Artemis",
    "HEC Triton", "HEC Oceanus", "HEC Nereus", "HEC Thetis", "HEC Proteus",
    "Green Star I", "Green Star II", "Green Star III", "Green Star IV", "Green Star V",
    "Green Collector I", "Green Collector II", "Green Collector III",
    "Green Titan I", "Green Titan II",
    "Eco Guardian I", "Eco Guardian II", "Eco Guardian III",
    "Sea Cleaner I", "Sea Cleaner II",
]

CLIENT_VESSEL_NAMES = [
    "MSC Fantasia", "Maersk Eindhoven", "CMA CGM Marco Polo", "Ever Given",
    "Costa Smeralda", "MSC Grandiosa", "Norwegian Epic", "Royal Princess",
    "Minerva Helen", "Torm Helene", "Stena Bulk", "Suezmax Explorer",
    "Pacific Voyager", "Atlantic Trader", "Nordic Spirit", "Mediterranean Sun",
    "Orient Express", "Golden Eagle", "Silver Cloud", "Diamond Princess",
    "Bulk Carrier Alpha", "Iron Ore Star", "Cape Fortune", "Baltic Pioneer",
    "Chemical Pioneer", "Stolt Tanker", "Odfjell Bow", "Navig8 Pride",
]

EQUIPMENT_MANUFACTURERS = {
    "Alfa Laval": ["ALCAP S 831", "ALCAP S 855", "ALDEC G3-95", "PX 90", "MAPX 207"],
    "GEA Westfalia": ["OSE 80-91-067", "OSD 60-91-067", "CSC 40-06-177", "CA 505"],
    "Flottweg": ["Z53-4/454", "Z73-4/441", "Sedicanter S4E", "Tricanter Z23"],
    "Andritz": ["D5LC30CP", "D7LC40CP", "D10LCC50CP"],
    "Pieralisi": ["FP 600 2RS", "Baby 1", "Mammoth"],
}


def seasonal_factor(date, port_code):
    """Return a seasonal multiplier for waste generation (0.5-1.5).
    Mediterranean ports peak in summer (cruise season), North European ports are steadier.
    """
    month = date.month
    is_med = port_code in ["PIR", "MLT", "GIB", "GEN", "MRS", "BCN", "ALG", "LIM", "ALE", "IST", "PSD", "TAN"]

    if is_med:
        # Mediterranean: peak Jul-Sep, low Jan-Feb
        factors = [0.60, 0.55, 0.70, 0.85, 1.00, 1.15, 1.35, 1.40, 1.30, 1.05, 0.80, 0.65]
    else:
        # North Europe (Hamburg, Rotterdam): steadier, slight dip in Dec
        factors = [0.90, 0.90, 0.95, 1.00, 1.05, 1.05, 1.10, 1.10, 1.05, 1.00, 0.95, 0.85]

    return factors[month - 1]


def temperature_for_location(date, facility_code):
    """Generate realistic ambient temperature based on location and month."""
    month = date.month
    # Monthly average temperatures (°C) — realistic for each location
    temps = {
        "PIR": [10, 10, 12, 16, 21, 26, 29, 29, 25, 20, 15, 11],
        "HAM": [1, 2, 5, 9, 14, 17, 19, 19, 15, 10, 6, 2],
        "GIB": [13, 13, 15, 17, 19, 23, 25, 26, 23, 20, 16, 14],
        "MLT": [12, 12, 14, 16, 20, 24, 27, 28, 25, 21, 17, 13],
    }
    base = temps.get(facility_code, temps["PIR"])[month - 1]
    # Add daily variation (±5°C) and intra-day (±3°C)
    return base + np.random.normal(0, 3)


def walther_viscosity(viscosity_40c, temperature_c):
    """Approximate viscosity at a given temperature using Walther equation.
    viscosity decreases exponentially with temperature.
    """
    # Simplified: viscosity halves for every ~15°C increase above 40°C
    if temperature_c <= 40:
        return viscosity_40c * (1.5 ** ((40 - temperature_c) / 15))
    else:
        return viscosity_40c * (0.5 ** ((temperature_c - 40) / 15))


def generate_imo_number():
    """Generate a valid-format IMO number (7 digits starting with 9)."""
    return f"9{np.random.randint(100000, 999999)}"


def haversine_distance_nm(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in nautical miles."""
    R = 3440.065  # Earth radius in nautical miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c
```

- [ ] **Step 2: Create ports.csv reference file**

```python
# Generate via script — content:
# port_code,port_name,country,latitude,longitude,avg_daily_vessel_calls,hec_market_share_pct,storage_capacity_m3,competitor_count
# PIR,Piraeus,GR,37.9475,23.637,85,75,25000,1
# HAM,Hamburg,DE,53.5461,9.966,120,40,45000,3
# GIB,Gibraltar,GI,36.1408,-5.3536,70,65,15000,2
# MLT,Valletta,MT,35.8989,14.5146,45,80,10000,1
# GEN,Genoa,IT,44.4056,8.9463,55,20,20000,3
# MRS,Marseille,FR,43.3462,5.32,60,15,18000,4
# BCN,Barcelona,ES,41.359,2.1685,65,10,22000,4
# RTM,Rotterdam,NL,51.9054,4.4661,140,8,50000,5
# ALG,Algeciras,ES,36.1272,-5.4427,50,55,12000,2
# LIM,Limassol,CY,34.6741,33.0379,30,60,8000,1
# ALE,Alexandria,EG,31.1842,29.8763,35,30,12000,3
# IST,Istanbul,TR,41.0053,28.977,90,15,30000,4
# PSD,Port Said,EG,31.2565,32.2841,40,25,10000,3
# TAN,Tangier,MA,35.785,-5.8029,45,35,10000,2
```

- [ ] **Step 3: Create equipment_master.csv reference file**

Generate 40 pieces of equipment across 4 facilities with realistic manufacturers, models, capacities.

- [ ] **Step 4: Create port_distance_matrix.csv reference file**

14×14 matrix with realistic sea distances (nm) between all port pairs, based on actual shipping routes (not straight line — account for coastlines, straits).

- [ ] **Step 5: Create fleet_master.csv reference file**

25 HEC vessels with names, DWT, class, home port, tank capacity, fuel consumption rates.

- [ ] **Step 6: Create offshore_platforms.csv reference file**

8 platforms across North Sea and Eastern Mediterranean with realistic operators, production rates, positions.

---

## Task 3: Separation Data Generator (UC1 — Source Tables)

**Files:**
- Create: `data/generators/separation_data_generator.py`

- [ ] **Step 1: Implement `generate_waste_reception_log()` function**

Generate ~15,000 rows covering 2022-01-01 to 2026-06-01. Each row represents a waste collection event. Must:
- Distribute across 4 facilities proportional to annual volume (PIR 45%, HAM 24%, GIB 18%, MLT 13%)
- Apply seasonal patterns (cruise season for Med ports)
- Generate realistic volumes per waste category (bilge water = small volumes, cargo residues = large)
- Correlate vessel type with waste category (tankers → cargo residues, cruise → bilge, bulk → sludge)
- Correlate vessel GT with waste volume (larger vessel → more waste, GT^0.7 relationship)
- Generate realistic acceptance fees (€15-50/m³ depending on waste difficulty)

- [ ] **Step 2: Implement `generate_laboratory_analysis()` function**

Generate ~45,000 rows (3 per reception: inlet, mid-process, output). Must:
- Ensure water + oil + solids ≈ 100% (±0.5% measurement error)
- Correlate properties with waste category (bilge = high water/low viscosity, sludge = high viscosity/metals)
- Output samples show improvement (less water in oil, less oil in water)
- Apply Walther equation for viscosity-temperature relationship
- HFO-burning vessels → higher vanadium, nickel, sulfur in waste
- Correlate sodium/chloride with seawater contamination (bilge > sludge)
- Flash point correlates with oil type (light products < 60°C = dangerous)

- [ ] **Step 3: Implement `generate_processing_batch()` function**

Generate ~12,000 rows. Must:
- Link to waste_reception_log (1:1 mostly, some combine 2-3 receptions)
- Calculate oil_recovery_yield_pct based on realistic formula:
  - Base yield from waste category (cargo residues 80-95%, sludge 40-70%, bilge 30-60%)
  - Modify by: viscosity (-), emulsion stability (-), temperature (+), operator experience (+), equipment condition (+), pre-treatment (+)
  - Add noise (±5%)
- Calculate energy_consumed based on volume × specific energy (affected by viscosity, temperature delta)
- Revenue = recovered_oil_volume × oil_price × quality_factor
- Quality pass rate ~85% overall, varies by waste type

- [ ] **Step 4: Implement `generate_process_sensor_data()` function**

Generate ~500,000 rows (sampled — not full 2M, to keep manageable). Must:
- 1-minute granularity during batch processing
- Sensor values physically consistent with batch parameters
- Centrifuge RPM varies by separator type (disc stack 5000-10000, decanter 2000-5000)
- Flow rate + RPM + temperature determine separation performance
- Add realistic sensor noise (±1-3% of reading)
- Vibration increases with running hours (equipment degradation signal)
- Bearing temperature correlates with load and age

- [ ] **Step 5: Implement `generate_weather_conditions()` function**

Generate ~140,000 rows (hourly × 4 facilities × 4 years). Must:
- Realistic temperature curves per location and season
- Mediterranean: dry summers, rainy winters
- Hamburg: cold winters, mild summers, frequent rain
- Wind patterns: stronger in winter, Meltemi in summer (Piraeus)

- [ ] **Step 6: Wire up main generation script with progress reporting**

```python
# In scripts/generate_data.py:
# Call each generator, save to data/raw/*.csv, print row counts and sample
```

---

## Task 4: Fleet Data Generator (UC2 — Source Tables)

**Files:**
- Create: `data/generators/fleet_data_generator.py`

- [ ] **Step 1: Implement `generate_port_waste_demand()` function**

Generate ~80,000 rows (daily × 14 ports × ~4 years). Must:
- Daily waste volumes proportional to vessel calls
- Vessel calls follow seasonal patterns (cruise season for Med)
- Larger ports (Rotterdam, Hamburg) have more vessel calls but lower HEC market share
- Storage fill percentage increases between HEC collections, resets after
- Waste breakdown (bilge/sludge/slops) varies by port (tanker ports → more slops)
- Weekend effect: slightly lower activity
- Year-over-year growth ~3-5% (shipping demand increase)

- [ ] **Step 2: Implement `generate_hec_fleet_status()` function**

Generate ~500,000 rows (every 4 hours × 25 vessels × 4 years). Must:
- Realistic vessel movements between ports (follow distance matrix)
- Status transitions: Loading → In Transit Laden → Discharging → In Transit Ballast → Loading
- Speed varies: laden 9-11 knots, ballast 10-13 knots
- Cargo level increases at collection ports, decreases at treatment facilities
- Maintenance periods: ~2 weeks/year per vessel (staggered)
- Fuel consumption proportional to speed³
- Position interpolation along realistic routes (not straight lines through land)

- [ ] **Step 3: Implement `generate_voyage_history()` function**

Generate ~8,000 rows. Must:
- Each voyage = departure → arrival (may include multiple collection stops)
- Revenue = sum of acceptance fees from collections on this voyage
- Cost = fuel (distance × consumption × fuel_price) + port_charges + crew
- Weather delays: more frequent in winter, North Sea worse than Med
- Multi-stop voyages: coastal vessels do 3-8 stops per voyage, large offshore vessels do 1
- Fuel consumed correlates with distance, speed, laden/ballast state

- [ ] **Step 4: Implement `generate_offshore_platform_data()` function**

Generate ~5,000 rows (daily × 8 platforms × ~2 years). Must:
- Production declines over time (natural field depletion)
- Water cut increases over time (reservoir aging — 10% → 80% over field life)
- Waste generation correlates with production × water cut
- Storage fills linearly between collections (every 30-60 days)
- Weather restrictions: North Sea platforms inaccessible in Beaufort > 5

- [ ] **Step 5: Implement `generate_maritime_weather()` function**

Generate ~24,000 rows (6-hourly × 6 zones × ~4 years). Must:
- Mediterranean calmer in summer, storms Oct-Mar
- North Sea rough in winter (avg Hs 2-4m), calmer Jun-Aug (Hs 1-2m)
- Gibraltar strait has strong currents (affects transit times)
- Realistic Beaufort/wave height correlation
- Operation feasibility based on vessel size thresholds

- [ ] **Step 6: Implement `generate_oil_market_data()` function**

Generate ~1,500 rows (daily × 4 years). Must:
- Brent crude: random walk with mean reversion around $75-85/bbl
- HFO/VLSFO/MGO prices correlated with Brent (spread relationships)
- Carbon credit price trending upward (€40 → €90 over 4 years)
- Shipping demand index: seasonal with economic cycle
- EUR/USD: range 0.95-1.15 with slow drift

---

## Task 5: Feature Engineering — Separation (UC1)

**Files:**
- Create: `features/separation_features.py`

- [ ] **Step 1: Implement Category A — Feedstock Complexity Features**

```python
def compute_feedstock_features(reception_df, lab_df):
    """
    Compute:
    - viscosity_temperature_ratio = viscosity_40c / (temperature_inlet + 273.15)
    - emulsion_difficulty_score = emulsion_layer_pct × (1 + viscosity_40c/1000) × stability_factor
    - oil_water_density_gap = 1000 - density_15c (approximation for oil/water gap)
    - solid_particle_settling_velocity = (d² × Δρ × g) / (18 × μ)  [Stokes' law]
    - waste_age_degradation = log(days_in_tank + 1) × (temperature_factor)
    - sulfur_to_oil_ratio = sulfur_pct / oil_content_pct
    - contamination_index = (vanadium + nickel + iron + sodium) / 1000
    - cat_fines_risk = 1 if aluminum_silicon_ppm > 60 else 0
    """
```

- [ ] **Step 2: Implement Category B — Process Optimization Features**

```python
def compute_process_features(batch_df, sensor_df, equipment_df):
    """
    Compute:
    - specific_energy_input = energy_consumed_kwh / total_volume_input_m3
    - g_force = (2*pi*rpm/60)^2 * bowl_radius_m
    - residence_time_actual = total_volume_input_m3 / feed_flow_rate_m3_hr * 60  (minutes)
    - chemical_to_emulsion_ratio = chemical_pump_rate / emulsion_layer_pct
    - temperature_vs_pour_point_margin = feed_temperature - pour_point
    - capacity_utilization = feed_flow_rate / equipment_capacity_m3_hr
    """
```

- [ ] **Step 3: Implement Category C — Temporal/Contextual Features**

```python
def compute_temporal_features(batch_df):
    """
    Compute:
    - hour_of_day, day_of_week, month (from timestamp)
    - rolling_yield_7d = 7-day moving average of yield per facility
    - rolling_energy_7d = 7-day moving average of specific energy
    - equipment_hours_since_service = running_hours - last_overhaul_hours
    - similar_batch_avg_yield = avg yield for same waste_subcategory in last 30 days
    - operator_avg_yield = operator's avg yield over last 90 days
    - ambient_heating_delta = process_temp - ambient_temp
    """
```

- [ ] **Step 4: Implement Category D — Interaction Features**

```python
def compute_interaction_features(features_df):
    """
    Compute:
    - viscosity_x_flow_rate = viscosity × feed_flow
    - solids_x_rpm = solids_content × centrifuge_rpm
    - water_x_temperature = water_content × (100 - temperature)
    - sulfur_x_volume = sulfur_pct × volume
    - age_x_emulsion = waste_age_degradation × emulsion_difficulty_score
    """
```

- [ ] **Step 5: Implement main pipeline function that joins all sources and outputs enriched dataset**

```python
def build_separation_feature_matrix(raw_data_dir, output_path):
    """Load all raw tables, join, compute all feature categories, save enriched CSV."""
```

---

## Task 6: Feature Engineering — Fleet (UC2)

**Files:**
- Create: `features/fleet_features.py`

- [ ] **Step 1: Implement Category E — Demand Prediction Features**

```python
def compute_demand_features(demand_df, weather_df, market_df):
    """
    Compute:
    - vessel_calls_7d_rolling, vessel_calls_30d_rolling
    - waste_per_vessel_call = waste_volume / vessel_calls_total
    - yoy_growth_pct = (current - same_period_last_year) / same_period_last_year
    - cruise_season_flag = 1 if month in [4..10] AND port is Mediterranean
    - port_congestion_proxy = vessel_calls / port_capacity_factor
    - days_since_last_collection (computed from demand history)
    - storage_fill_rate_m3_day = diff(current_storage_fill_pct) × capacity
    - weather_window_probability = % of next 7 days with operation_feasibility == 'Go'
    """
```

- [ ] **Step 2: Implement Category F — Route Optimization Features**

```python
def compute_route_features(fleet_df, demand_df, distance_df, weather_df):
    """
    Compute:
    - vessel_proximity_to_demand_nm = haversine(vessel_pos, port_pos)
    - vessel_available_capacity_m3 = tank_capacity - current_cargo
    - urgency_score = storage_fill_pct × (1 / max(days_until_full, 1))
    - collection_efficiency_m3_per_nm = predicted_volume / distance
    - fuel_cost_per_m3_collected = (distance × consumption × fuel_price) / volume
    - multi_stop_opportunity = count ports within 100nm with demand > threshold
    - weather_risk_enroute = max wave height on route in next 48h
    - contract_obligation_days = SLA deadline - current_date
    - vessel_fuel_autonomy_days = fuel_remaining / daily_consumption
    """
```

- [ ] **Step 3: Implement Category G — Economic Features**

```python
def compute_economic_features(demand_df, market_df, distance_df):
    """
    Compute:
    - expected_revenue_per_voyage = predicted_volume × avg_acceptance_fee + recovery_value
    - voyage_cost_estimate = fuel_cost + port_charges + crew_cost
    - expected_margin = revenue - cost
    - oil_price_trend_7d = pct_change(brent, 7d)
    - market_share_risk = 1 if competitor_presence > 2 AND days_since_collection > 5
    """
```

- [ ] **Step 4: Implement main pipeline function for fleet feature matrix**

```python
def build_fleet_feature_matrix(raw_data_dir, output_path):
    """Load all raw tables, join, compute all feature categories, save enriched CSV."""
```

---

## Task 7: ML Models — Separation (UC1)

**Files:**
- Create: `models/separation/yield_predictor.py`
- Create: `models/separation/parameter_optimizer.py`
- Create: `models/separation/quality_classifier.py`

- [ ] **Step 1: Implement Model 1A — Yield Prediction (XGBoost Regressor)**

```python
"""
Target: oil_recovery_yield_pct
Features: All Category A, B, C, D features
Split: 80/20 train/test, time-based split (train on 2022-2025, test on 2025-2026)
Metrics: RMSE, MAE, R²
Output: trained model saved to models/trained/yield_predictor.joblib
Also save: feature_importances, evaluation_metrics, predictions_vs_actuals
"""
```

- [ ] **Step 2: Implement Model 1B — Parameter Optimizer (Bayesian Optimization)**

```python
"""
Uses trained yield predictor as objective function.
Optimizes: feed_rate, temperature, rpm, chemical_dose, residence_time, backpressure
Subject to constraints:
  - Equipment limits (max RPM, max temp, max flow)
  - Safety: flash_point > process_temp (no ignition risk)
  - Environmental: predicted water_output_oil_ppm < 15
  - Time: processing_time < deadline
  - Budget: energy_cost < budget
Uses scipy.optimize.minimize with bounds, or skopt BayesSearchCV
Output: function that takes batch characteristics → returns optimal parameters
"""
```

- [ ] **Step 3: Implement Model 1C — Quality Risk Classifier (Random Forest)**

```python
"""
Target: quality_pass (binary) + quality_failure_reason (multi-class)
Features: Same as Model 1A + predicted yield from 1A
Split: Same time-based split
Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
Output: trained model + classification_report saved
"""
```

---

## Task 8: ML Models — Fleet (UC2)

**Files:**
- Create: `models/fleet/demand_forecaster.py`
- Create: `models/fleet/route_optimizer.py`
- Create: `models/fleet/profitability_model.py`

- [ ] **Step 1: Implement Model 2A — Demand Forecasting (XGBoost + seasonal decomposition)**

```python
"""
Target: waste_volume_collected_m3 per port per day, 7-day ahead
Algorithm: XGBoost with lag features + seasonal indicators (simpler than Prophet for demo)
Features: Category E features + port-specific seasonality + oil market
Split: Time-based (train 2022-2025, test 2025-2026)
Metrics: MAPE, RMSE, prediction intervals (quantile regression)
Output: trained model + forecasts for each port saved
"""
```

- [ ] **Step 2: Implement Model 2B — Route Optimizer (Greedy heuristic + scoring)**

```python
"""
Not full MILP (too complex for demo), but smart greedy assignment:
1. Score all (vessel, port) pairs using urgency × efficiency × feasibility
2. Assign highest-scoring pair, update vessel state
3. Repeat until all urgent demands assigned or fleet exhausted
Constraints: vessel capacity, fuel autonomy, weather feasibility, SLA deadlines
Output: assignment_plan DataFrame with vessel_id, port_sequence, ETA, expected_volume
"""
```

- [ ] **Step 3: Implement Model 2C — Voyage Profitability (XGBoost Regressor)**

```python
"""
Target: voyage_margin_eur
Features: Category F + G features
Split: Time-based
Metrics: RMSE, MAE, R²
Output: trained model + feature importances
"""
```

---

## Task 9: Data Generation & Model Training Script

**Files:**
- Create: `scripts/generate_data.py`
- Create: `scripts/engineer_features.py`
- Create: `scripts/train_models.py`

- [ ] **Step 1: Create generate_data.py — orchestrates all data generators**

```python
"""
Calls:
1. separation_data_generator.generate_all() → saves 6 CSVs to data/raw/
2. fleet_data_generator.generate_all() → saves 6 CSVs to data/raw/
Prints progress, row counts, and data quality checks.
"""
```

- [ ] **Step 2: Create engineer_features.py — runs both feature pipelines**

```python
"""
Calls:
1. separation_features.build_separation_feature_matrix() → saves to data/features/
2. fleet_features.build_fleet_feature_matrix() → saves to data/features/
Prints feature counts, null checks, distribution summaries.
"""
```

- [ ] **Step 3: Create train_models.py — trains all 6 models**

```python
"""
Calls:
1. yield_predictor.train_and_save()
2. quality_classifier.train_and_save()
3. parameter_optimizer.build_optimizer()
4. demand_forecaster.train_and_save()
5. route_optimizer.build_optimizer()
6. profitability_model.train_and_save()
Prints metrics for each model.
"""
```

---

## Task 10: Streamlit App — Main Entry & Config

**Files:**
- Create: `app/config.py`
- Create: `app/main.py`

- [ ] **Step 1: Create config.py with paths and display settings**

```python
"""
DATA_RAW_DIR, DATA_FEATURES_DIR, MODELS_DIR paths
Color schemes, chart defaults, page layout config
Column descriptions dictionary (for tooltips in UI)
"""
```

- [ ] **Step 2: Create main.py — Streamlit entry point**

```python
"""
st.set_page_config(page_title="HEC AI Platform", layout="wide")
Sidebar with:
  - HEC logo/title
  - Navigation description
  - Data generation status (files exist?)
  - Quick stats (total batches, vessels, etc.)
Main page: overview dashboard with key KPIs from both use cases
"""
```

---

## Task 11: Streamlit — Separation Control Page (Tab 1)

**Files:**
- Create: `app/pages/1_separation_control.py`
- Create: `app/components/pipeline_viewer.py`
- Create: `app/components/charts.py`

- [ ] **Step 1: Create pipeline_viewer.py — reusable Source→FE→Model→Result component**

```python
"""
Renders a 4-step expandable pipeline:
1. st.expander("1. Source Data") → show raw data sample + stats
2. st.expander("2. Feature Engineering") → show transformations + enriched columns
3. st.expander("3. ML Model") → show model info, training metrics
4. st.expander("4. Results") → show predictions/optimizations
Each step has a progress indicator and data preview.
"""
```

- [ ] **Step 2: Create charts.py — plotting utilities**

```python
"""
Functions for:
- correlation_heatmap(df, columns)
- distribution_plot(df, column, by_category)
- actual_vs_predicted_scatter(y_true, y_pred)
- feature_importance_bar(importances, feature_names, top_n)
- time_series_with_forecast(actual, predicted, dates)
- residual_plot(y_true, y_pred)
- shap_summary_plot(shap_values, features)  # if shap available
All using Plotly for interactivity.
"""
```

- [ ] **Step 3: Implement View 1.1 — Source Data Explorer section**

```python
"""
- Filterable data table (waste_reception_log + lab_analysis joined)
- Selectbox: filter by facility, waste category, date range
- Distribution charts of key properties (viscosity, water%, oil%, density) by waste category
- Correlation matrix heatmap of numeric lab columns
- Time series: incoming volume per facility per month
"""
```

- [ ] **Step 4: Implement View 1.2 — Feature Engineering Pipeline section**

```python
"""
- Select a batch from dropdown
- Show raw columns for that batch (left side)
- Show computed features (right side) with formulas
- Feature importance ranking from trained model (bar chart)
- Feature distribution histograms with outlier markers
"""
```

- [ ] **Step 5: Implement View 1.3 — Model Performance section**

```python
"""
- Actual vs. Predicted yield scatter (with R² annotation)
- Residual distribution histogram
- Feature importance (top 20) bar chart
- Error by waste category (box plot of |residuals| per category)
- Metrics table: RMSE, MAE, R², MAPE per facility
"""
```

- [ ] **Step 6: Implement View 1.4 — Optimization Engine section**

```python
"""
- Dropdown: select a batch (or "New Batch" with manual inputs)
- If new: sliders for waste properties (viscosity, water%, oil%, temp, etc.)
- Show predicted yield with current parameters
- Show AI-recommended optimal parameters (table with before/after)
- Slider overrides: user can adjust each parameter, see yield change live
- Revenue/cost impact: show € difference between current and optimal
- Quality risk traffic light: Green/Amber/Red based on classifier
"""
```

---

## Task 12: Streamlit — Fleet Routing Page (Tab 2)

**Files:**
- Create: `app/pages/2_fleet_routing.py`
- Create: `app/components/map_viewer.py`

- [ ] **Step 1: Create map_viewer.py — port/fleet map component**

```python
"""
Using pydeck or folium:
- Show all 14 ports as colored circles (color = demand level / urgency)
- Show HEC fleet vessels as ship icons at current positions
- Draw route lines between assigned vessel→port pairs
- Color code: green=available, yellow=in transit, red=at capacity
- Popup on click: port stats / vessel stats
"""
```

- [ ] **Step 2: Implement View 2.1 — Source Data Explorer section**

```python
"""
- Map with ports and fleet positions
- Date selector to view historical state
- Time series: waste volume per port (stacked area chart)
- Oil market overlay (Brent price line on secondary axis)
- Port comparison table with key metrics
"""
```

- [ ] **Step 3: Implement View 2.2 — Demand Forecast Pipeline section**

```python
"""
- Select a port from dropdown
- Show raw demand history (source data)
- Show computed features (rolling averages, seasonal flags, etc.)
- Show forecast vs. actual (line chart with confidence bands)
- Accuracy metrics per port: MAPE, RMSE
- Seasonal pattern decomposition visualization
"""
```

- [ ] **Step 4: Implement View 2.3 — Fleet Optimization Dashboard section**

```python
"""
- Current fleet status table (vessel, position, cargo%, status)
- Urgency ranking: ports sorted by urgency_score
- Recommended assignments: vessel → port with expected arrival, cost, revenue
- KPI cards: fleet utilization %, avg empty miles, SLA compliance %
- What-if: slider to remove 1-2 vessels, see impact on coverage
"""
```

- [ ] **Step 5: Implement View 2.4 — Model Performance section**

```python
"""
- Forecast accuracy by port (bar chart of MAPE per port)
- Demand forecast time series overlay (actual vs predicted for selected port)
- Profitability heatmap: port × month matrix colored by avg margin
- Cost breakdown pie chart: fuel %, port %, crew %, other %
"""
```

---

## Task 13: Integration, Testing & Polish

**Files:**
- Modify: `app/main.py`
- All files: final integration check

- [ ] **Step 1: Run full pipeline end-to-end**

```bash
cd C:\SAPDevelop\meli
python scripts/generate_data.py
python scripts/engineer_features.py
python scripts/train_models.py
```

Verify: All CSVs created in data/raw/ and data/features/, all .joblib models in models/trained/

- [ ] **Step 2: Launch Streamlit and verify all pages load**

```bash
streamlit run app/main.py
```

Verify: No errors, all pages render, data displays correctly.

- [ ] **Step 3: Test Separation Control page interactions**

- Filter by facility, waste category
- Select a batch, verify feature engineering display
- Check model metrics display
- Test optimization engine with slider inputs

- [ ] **Step 4: Test Fleet Routing page interactions**

- Verify map renders with ports and vessels
- Select different dates, verify data changes
- Select port for forecast, verify chart
- Check fleet optimization recommendations

- [ ] **Step 5: Add column descriptions/tooltips throughout the app**

Every data table should show column explanations on hover or in an info expander, so users understand what each field means and where it comes from (which source system).

- [ ] **Step 6: Final smoke test — verify the Source→FE→Model→Result pipeline is clear in every view**

Walk through each view confirming the 4-step logic is visible and understandable.
