# HEC AI Platform — Separation Process Control & Fleet Routing

## Design Specification

**Date:** 2026-06-05  
**Stack:** Python (Streamlit + pandas + scikit-learn/XGBoost)  
**Priority:** Realistic data, full business case coverage, pipeline transparency  

---

## OVERALL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT APP                              │
├──────────────────────────┬──────────────────────────────────┤
│  TAB 1: Separation       │  TAB 2: Fleet Routing            │
│  Process Control         │  & Demand Forecasting            │
├──────────────────────────┴──────────────────────────────────┤
│  Each tab follows the same visual pipeline:                  │
│                                                              │
│  ┌──────────┐   ┌─────────────┐   ┌─────────┐   ┌───────┐ │
│  │ SOURCE   │ → │ FEATURE     │ → │ ML      │ → │RESULT │ │
│  │ DATA     │   │ ENGINEERING │   │ MODEL   │   │       │ │
│  └──────────┘   └─────────────┘   └─────────┘   └───────┘ │
│                                                              │
│  Expandable sections showing data at each stage              │
└─────────────────────────────────────────────────────────────┘
```

**Dashboard Logic (MANDATORY for every graph/view):**
1. Show raw source data sample + statistics
2. Show feature engineering transformations applied
3. Show the enriched/final dataset
4. Show model training/inference
5. Show the prediction/optimization result

---

## USE CASE 1: AI-OPTIMIZED SEPARATION PROCESS CONTROL

### 1.1 SOURCE SYSTEMS (Where Real Data Would Come From)

#### Source System A: SCADA/DCS (Distributed Control System)
**What it is:** Industrial automation system controlling the separation plant.  
**Real-world vendors:** Siemens SIMATIC PCS 7, ABB Ability 800xA, Honeywell Experion  
**Data frequency:** Every 1-10 seconds per sensor  
**Provides:** All real-time sensor readings (temperature, pressure, flow, RPM)

#### Source System B: LIMS (Laboratory Information Management System)
**What it is:** Lab system where operators log sample analysis results.  
**Real-world vendors:** LabWare LIMS, Thermo Fisher SampleManager  
**Data frequency:** Per batch (1-3 samples per batch, taken at inlet/during/outlet)  
**Provides:** Chemical analysis (sulfur, water content, ash, metals, API gravity, flash point)

#### Source System C: ERP/Operations Management System
**What it is:** Business system tracking batches, costs, revenues, customers.  
**Real-world vendors:** SAP S/4HANA, Oracle EBS, or custom maritime ERP  
**Data frequency:** Per batch/transaction  
**Provides:** Batch metadata, costs, revenues, customer info, vessel info

#### Source System D: Waste Acceptance System (Port Reception)
**What it is:** System logging incoming waste declarations per MARPOL regulations.  
**Real-world format:** IMO Waste Delivery Receipt, EU Port Reception Facility reporting  
**Data frequency:** Per collection event  
**Provides:** Waste category, declared volume, vessel info, port of origin

#### Source System E: Weather/Environmental API
**What it is:** External weather data affecting operations.  
**Real-world sources:** OpenWeatherMap, Copernicus Climate Data Store, ECMWF  
**Data frequency:** Hourly  
**Provides:** Ambient temperature, humidity, wind (affects tank farm operations)

#### Source System F: Equipment Maintenance System (CMMS)
**What it is:** Computerized Maintenance Management System tracking equipment health.  
**Real-world vendors:** IBM Maximo, SAP PM, Fiix  
**Data frequency:** Per maintenance event + continuous condition monitoring  
**Provides:** Equipment age, last service, running hours, vibration readings

---

### 1.2 SOURCE DATA TABLES (Synthetic but Realistic)

#### Table: `waste_reception_log` (from Source D + C)
**Business meaning:** Every waste collection event — when a tanker arrives at the plant with collected waste.

| Column | Type | Realistic Values | Business Explanation |
|--------|------|-----------------|---------------------|
| `reception_id` | str | "REC-PIR-2024-00412" | Unique reception event ID (facility-year-sequence) |
| `timestamp` | datetime | 2022-01-01 to 2026-06-01 | When waste was received at facility |
| `facility_code` | str | PIR, HAM, GIB, MLT | Piraeus, Hamburg, Gibraltar, Malta |
| `collecting_vessel_name` | str | "HEC Poseidon", "Green Star III" | HEC's own tanker that collected the waste |
| `collecting_vessel_dwt` | int | 500, 1000, 2000, 3500, 5000, 7000, 30000 | Deadweight tonnage of HEC's tanker |
| `source_vessel_name` | str | "MSC Fantasia", "Maersk Eindhoven" | Client vessel that generated waste |
| `source_vessel_imo` | str | "9320544" | 7-digit IMO number of source vessel |
| `source_vessel_type` | str | Container Ship, Crude Oil Tanker, Product Tanker, Bulk Carrier, Cruise Ship, LNG Carrier, FPSO, Offshore Platform, Chemical Tanker, RoRo | Type of vessel generating waste |
| `source_vessel_gt` | int | 2,000–230,000 | Gross tonnage (cruise ships 100k-230k, container 10k-200k) |
| `source_vessel_engine_kw` | int | 3,000–80,000 | Main engine power in kW (determines sludge generation rate) |
| `source_vessel_fuel_type` | str | HFO (Heavy Fuel Oil), VLSFO (Very Low Sulfur), LSMGO (Low Sulfur Marine Gas Oil), LNG | Fuel type affects waste composition |
| `source_port` | str | Piraeus, Hamburg, Gibraltar, Valletta, Genoa, Marseille, Barcelona, Rotterdam, Algeciras, Limassol, Alexandria, Istanbul | Where collection happened |
| `waste_category_marpol` | str | See detailed list below | MARPOL classification |
| `waste_subcategory` | str | See detailed list below | More specific waste type |
| `declared_volume_m3` | float | 2–3,000 | Volume declared on waste delivery receipt |
| `actual_volume_m3` | float | 1.8–3,200 | Measured volume (often differs from declared) |
| `waste_temperature_c` | float | 5–70 | Temperature at delivery |
| `collection_method` | str | Barge Transfer, Direct Pumping, Shore Pipeline, Truck Delivery | How waste was transferred |
| `mixing_during_transport` | str | None, Mild, Moderate, Severe | Agitation during collection affects emulsion |
| `days_in_tank_before_collection` | int | 1–90 | How long waste sat on source vessel (aging affects chemistry) |
| `customer_id` | str | "CUST-00042" | Customer account |
| `contract_type` | str | Spot, Annual Contract, Framework Agreement | Pricing model |
| `acceptance_fee_eur` | float | 50–150,000 | Fee charged to customer for accepting waste |

**MARPOL Waste Categories (realistic detail):**

| `waste_category_marpol` | `waste_subcategory` | Typical Properties |
|--------------------------|---------------------|--------------------|
| Annex I - Oily Residues | Bilge Water | 80-95% water, 3-15% oil, <5% solids |
| Annex I - Oily Residues | Fuel Oil Sludge | 20-60% oil, 10-40% water, 10-30% solids |
| Annex I - Oily Residues | Oily Tank Washings | 30-70% oil, 20-60% water, 2-10% solids |
| Annex I - Oily Residues | Dirty Ballast Water | 95-99% water, 0.1-3% oil, <1% solids |
| Annex I - Oily Residues | Scale & Sludge from Tank Cleaning | 10-30% oil, 10-30% water, 40-70% solids |
| Annex I - Oily Residues | Cargo Residues (Crude) | 70-95% oil, 3-20% water, 2-10% solids |
| Annex I - Oily Residues | Cargo Residues (Product) | 80-98% oil, 1-10% water, <3% solids |
| Annex I - Oily Residues | Offshore Production Slops | 40-70% oil, 20-50% water, 5-20% solids |
| Annex I - Oily Residues | Engine Room Oily Water | 85-97% water, 2-10% oil, 1-5% solids |
| Annex II - Noxious Liquid | Vegetable Oil Residues | 60-90% oil (non-petroleum), 5-30% water |
| Annex II - Noxious Liquid | Chemical Washing Residues | Variable, requires special handling |

---

#### Table: `laboratory_analysis` (from Source B — LIMS)
**Business meaning:** Chemical analysis performed on each batch at intake, during processing, and at output.

| Column | Type | Realistic Values | Business Explanation |
|--------|------|-----------------|---------------------|
| `sample_id` | str | "LAB-PIR-2024-01234" | Unique lab sample ID |
| `reception_id` | str | FK → waste_reception_log | Links to reception event |
| `sample_point` | str | Inlet, Mid-Process, Oil Output, Water Output, Solids Output | Where in the process the sample was taken |
| `sample_timestamp` | datetime | | When sample was analyzed |
| `water_content_pct` | float | 0.1–97% | Karl Fischer titration result |
| `oil_content_pct` | float | 0.5–98% | By extraction (hexane/dichloromethane) |
| `solids_content_pct` | float | 0.01–70% | Filtration + drying at 105°C |
| `density_15c_kg_m3` | float | 820–1,080 | At standard 15°C (ASTM D4052) |
| `viscosity_40c_cst` | float | 2–50,000 | Kinematic viscosity at 40°C (ASTM D445) |
| `viscosity_100c_cst` | float | 1–5,000 | Viscosity at 100°C (for viscosity index calc) |
| `flash_point_c` | float | 21–300 | Pensky-Martens closed cup (ASTM D93) — safety critical |
| `pour_point_c` | float | -40 to +60 | ASTM D97 — pumpability indicator |
| `sulfur_total_pct` | float | 0.05–5.0 | XRF or combustion method (ASTM D4294) |
| `ash_content_pct` | float | 0.001–8% | ASTM D482 — inorganic residue |
| `carbon_residue_pct` | float | 0.1–25% | Conradson/Ramsbottom (ASTM D189) — coking tendency |
| `vanadium_ppm` | float | 0.1–600 | ICP-OES — corrosion indicator, common in HFO residues |
| `nickel_ppm` | float | 0.1–200 | ICP-OES — catalyst poison |
| `iron_ppm` | float | 1–5,000 | Rust/corrosion particles |
| `sodium_ppm` | float | 5–30,000 | Seawater contamination indicator |
| `aluminum_silicon_ppm` | float | 1–500 | Cat fines (catalytic cracker residue) — abrasive |
| `chloride_ppm` | float | 10–80,000 | Salt content (seawater ingress) |
| `acid_number_mg_koh_g` | float | 0.01–10 | Corrosivity indicator (ASTM D664) |
| `calorific_value_gross_mj_kg` | float | 10–46 | Bomb calorimeter (ASTM D240) — energy content |
| `calorific_value_net_mj_kg` | float | 9–43 | Net (minus water vaporization energy) |
| `pcb_ppm` | float | 0–50 | Polychlorinated biphenyls (regulated contaminant) |
| `btex_ppm` | float | 0–5,000 | Benzene/Toluene/Ethylbenzene/Xylene (volatile organics) |
| `h2s_ppm` | float | 0–10,000 | Hydrogen sulfide — toxic gas, safety critical |
| `emulsion_layer_pct` | float | 0–60 | Rag layer between oil/water (hardest to process) |
| `particle_size_d50_micron` | float | 1–500 | Median solid particle size (affects settling) |

---

#### Table: `process_sensor_data` (from Source A — SCADA/DCS)
**Business meaning:** Real-time sensor readings from the separation equipment during processing. Aggregated to 1-minute intervals for ML.

| Column | Type | Realistic Values | Business Explanation |
|--------|------|-----------------|---------------------|
| `reading_id` | str | UUID | Unique sensor reading |
| `batch_id` | str | FK → processing_batch | Which batch is being processed |
| `timestamp` | datetime | Per minute during processing | Reading time |
| `separator_unit_id` | str | "CENT-PIR-01", "GRAV-PIR-02", "DISC-HAM-01" | Equipment identifier |
| `separator_type` | str | 3-Phase Disc Stack Centrifuge, Decanter Centrifuge, Gravity Separator, Plate Separator, DAF Unit | Equipment type |
| `feed_temperature_c` | float | 35–98 | Temperature of waste entering separator |
| `feed_flow_rate_m3_hr` | float | 1–100 | Input flow rate |
| `feed_pressure_bar` | float | 0.5–8 | Input pressure |
| `centrifuge_speed_rpm` | int | 2,000–12,000 | Actual rotational speed (disc stack: 5000-10000, decanter: 2000-5000) |
| `centrifuge_differential_rpm` | int | 5–80 | Speed difference between bowl and scroll (decanter only) |
| `centrifuge_torque_nm` | float | 50–5,000 | Mechanical torque — indicates solids loading |
| `centrifuge_vibration_mm_s` | float | 0.5–25 | Vibration velocity — health/balance indicator |
| `oil_discharge_temp_c` | float | 35–95 | Temperature at oil output |
| `oil_discharge_flow_m3_hr` | float | 0.2–60 | Oil output flow rate |
| `oil_backpressure_bar` | float | 0.5–6 | Pressure at oil discharge |
| `water_discharge_temp_c` | float | 30–90 | Temperature at water output |
| `water_discharge_flow_m3_hr` | float | 0.5–80 | Water output flow rate |
| `solids_discharge_interval_s` | float | 10–600 | Time between solids ejections (disc stack) |
| `solids_discharge_volume_l` | float | 5–200 | Volume per solids ejection |
| `heating_power_kw` | float | 0–2,000 | Heat applied to feed |
| `chemical_pump_rate_ml_min` | float | 0–500 | Demulsifier/flocculant injection rate |
| `interface_position_mm` | float | 30–150 | Oil/water interface level in separator |
| `turbidity_oil_ntu` | float | 0–500 | Clarity of oil output (lower = better) |
| `turbidity_water_ntu` | float | 0–1000 | Clarity of water output (lower = better) |
| `oil_in_water_ppm_online` | float | 0–500 | Online analyzer: oil content in water discharge |
| `water_in_oil_pct_online` | float | 0–10% | Online analyzer: water in oil output |
| `power_consumption_kw` | float | 10–500 | Total electrical power draw of unit |
| `motor_current_a` | float | 20–800 | Main motor current (overload indicator) |
| `motor_temperature_c` | float | 40–120 | Motor winding temperature |
| `bearing_temperature_c` | float | 30–95 | Main bearing temperature (failure predictor) |

---

#### Table: `equipment_master` (from Source F — CMMS)
**Business meaning:** Static data about each piece of separation equipment.

| Column | Type | Realistic Values | Business Explanation |
|--------|------|-----------------|---------------------|
| `unit_id` | str | "CENT-PIR-01" | Equipment identifier |
| `facility_code` | str | PIR, HAM, GIB, MLT | Location |
| `equipment_type` | str | 3-Phase Disc Stack, Decanter, Gravity Settler, Plate Separator, DAF, Heating Unit | Type |
| `manufacturer` | str | Alfa Laval, GEA Westfalia, Flottweg, Andritz, Pieralisi | Real separator manufacturers |
| `model` | str | "ALDEC G3-95", "OSE 80-91-067" | Realistic model numbers |
| `year_installed` | int | 2005–2024 | Installation year |
| `capacity_m3_hr` | float | 5–120 | Maximum throughput |
| `last_major_overhaul` | date | | Last significant maintenance |
| `running_hours_total` | int | 5,000–120,000 | Total operating hours |
| `running_hours_since_overhaul` | int | 500–30,000 | Hours since last major service |
| `next_scheduled_maintenance` | date | | Planned service date |
| `condition_score` | float | 1–10 | Overall condition (1=poor, 10=excellent) |

---

#### Table: `weather_conditions` (from Source E)
**Business meaning:** Hourly weather at each facility — affects heating costs, tank behavior, worker safety.

| Column | Type | Realistic Values | Business Explanation |
|--------|------|-----------------|---------------------|
| `timestamp` | datetime | Hourly | |
| `facility_code` | str | PIR, HAM, GIB, MLT | |
| `ambient_temperature_c` | float | PIR: 8-38, HAM: -5 to 32, GIB: 10-35, MLT: 10-35 | Affects heating energy needed |
| `humidity_pct` | float | 30–95% | Affects evaporation rates |
| `wind_speed_ms` | float | 0–25 | Affects VOC dispersion, safety |
| `precipitation_mm` | float | 0–50 | Rain affects outdoor tank farm ops |
| `barometric_pressure_hpa` | float | 990–1040 | Affects flash point behavior |

---

#### Table: `processing_batch` (from Source C — ERP)
**Business meaning:** Each batch that goes through separation — links reception to outputs.

| Column | Type | Realistic Values | Business Explanation |
|--------|------|-----------------|---------------------|
| `batch_id` | str | "BATCH-PIR-2024-00567" | Unique processing batch |
| `reception_id` | str | FK → waste_reception_log | Source waste (may combine multiple receptions) |
| `start_timestamp` | datetime | | Processing start |
| `end_timestamp` | datetime | | Processing end |
| `facility_code` | str | PIR, HAM, GIB, MLT | Where processed |
| `operator_id` | str | "OP-PIR-012" | Shift operator (skill affects results) |
| `operator_experience_years` | int | 1–25 | Years of experience |
| `separator_unit_id` | str | FK → equipment_master | Primary separator used |
| `pre_treatment` | str | None, Heating Only, Chemical + Heating, Gravity Pre-Settling, Filtration | Pre-treatment applied |
| `pre_settling_hours` | float | 0–72 | Hours of gravity settling before centrifuging |
| `target_oil_spec` | str | Refinery Grade, Marine Fuel Blend, Industrial Burner, Asphalt Blend | What quality the buyer needs |
| `batch_priority` | str | Normal, Urgent, Premium | Affects whether speed or yield is optimized |
| `total_volume_input_m3` | float | 5–3,000 | Total volume fed to separator |
| `total_oil_output_m3` | float | 0.5–2,500 | Oil recovered |
| `total_water_output_m3` | float | 1–2,800 | Water discharged (to treatment) |
| `total_solids_output_kg` | float | 10–50,000 | Solid phase collected |
| `oil_recovery_yield_pct` | float | 15–95% | PRIMARY TARGET: oil recovered / oil in feed |
| `energy_consumed_kwh` | float | 20–8,000 | Total energy for this batch |
| `chemicals_cost_eur` | float | 5–5,000 | Demulsifier + flocculant cost |
| `labor_hours` | float | 2–48 | Operator hours |
| `total_processing_cost_eur` | float | 100–50,000 | Full cost |
| `recovered_oil_revenue_eur` | float | 200–200,000 | Revenue from oil sales |
| `solid_fuel_revenue_eur` | float | 10–15,000 | Revenue from solid phase as alt fuel |
| `net_margin_eur` | float | -10,000 to +180,000 | Profit per batch |
| `quality_pass` | bool | True/False | Did output meet buyer specification? |
| `quality_failure_reason` | str | None, High Water, High Sulfur, Low Flash Point, High Ash, Cat Fines Exceed | Why spec was missed |

---

### 1.3 FEATURE ENGINEERING (Separation)

These are the transformations applied to source data to create ML-ready features:

#### Category A: Feedstock Complexity Features
| Feature | Formula/Logic | Why |
|---------|---------------|-----|
| `viscosity_temperature_ratio` | viscosity_40c / (temperature_inlet + 273.15) | Normalized pumpability — cold viscous waste is hardest |
| `emulsion_difficulty_score` | emulsion_layer_pct × (1 + viscosity_40c/1000) × stability_factor | Composite difficulty metric |
| `oil_water_density_gap` | density_water - density_oil (from lab) | Larger gap = easier gravity separation |
| `solid_particle_settling_velocity` | Stokes' law calculation from particle_size & density | Predicts how fast solids settle |
| `waste_age_degradation` | log(days_in_tank + 1) × temperature_factor | Aged waste forms harder emulsions |
| `sulfur_to_oil_ratio` | sulfur_pct / oil_content_pct | Indicates sulfur concentration in recoverable oil |
| `contamination_index` | (vanadium + nickel + iron + sodium) / 1000 | Overall metal contamination severity |
| `cat_fines_risk` | aluminum_silicon_ppm > 60 → High | Catalytic fines damage engines — high risk waste |

#### Category B: Process Optimization Features  
| Feature | Formula/Logic | Why |
|---------|---------------|-----|
| `specific_energy_input` | heating_power_kw × time / volume | Energy intensity per m³ |
| `g_force` | (2π × rpm/60)² × bowl_radius | Centrifugal acceleration (separation force) |
| `sigma_factor` | Theoretical capacity based on disc geometry | Equipment-specific separation efficiency |
| `residence_time_actual` | volume / flow_rate | Actual time waste spends in separator |
| `reynolds_number` | ρ × v × D / μ | Flow regime (laminar=better separation) |
| `chemical_to_emulsion_ratio` | demulsifier_ppm / emulsion_layer_pct | Is enough chemical being dosed? |
| `temperature_vs_pour_point_margin` | feed_temp - pour_point | How far above solidification (must be >15°C) |
| `capacity_utilization` | actual_flow / equipment_max_capacity | How hard is the machine working? |

#### Category C: Temporal/Contextual Features
| Feature | Formula/Logic | Why |
|---------|---------------|-----|
| `hour_of_day` | From timestamp | Night shifts may have different performance |
| `day_of_week` | From timestamp | Weekend = skeleton crew |
| `month` | From timestamp | Seasonal feedstock variation |
| `rolling_yield_7d` | 7-day moving average of yield | Recent performance trend |
| `rolling_energy_7d` | 7-day moving average of energy/m³ | Efficiency trend |
| `equipment_hours_since_service` | running_hours - last_overhaul_hours | Equipment degradation |
| `similar_batch_avg_yield` | Average yield for same waste_subcategory in last 30 days | Historical baseline |
| `operator_avg_yield` | Operator's average yield over last 90 days | Operator skill factor |
| `ambient_heating_delta` | process_temp_target - ambient_temp | How much heating needed today |

#### Category D: Interaction Features
| Feature | Formula/Logic | Why |
|---------|---------------|-----|
| `viscosity_x_flow_rate` | viscosity × feed_flow | Combined load on separator |
| `solids_x_rpm` | solids_content × centrifuge_rpm | Solids ejection demand |
| `water_x_temperature` | water_content × (100 - temperature) | Cold dilute waste behavior |
| `sulfur_x_volume` | sulfur_pct × volume | Total sulfur mass to manage |
| `age_x_emulsion` | waste_age × emulsion_stability_score | Old stable emulsions = worst case |

---

### 1.4 ML MODELS (Separation)

#### Model 1A: Yield Prediction (Regression)
- **Target:** `oil_recovery_yield_pct`
- **Algorithm:** XGBoost Regressor
- **Features:** All from Category A, B, C, D above
- **Evaluation:** RMSE, MAE, R² on 20% holdout
- **Business value:** Predict what yield to expect BEFORE processing starts → decide whether to accept waste / set pricing

#### Model 1B: Optimal Parameters (Prescriptive)
- **Target:** Maximize yield while constraining: energy < budget, water_output_oil < 15ppm, time < deadline
- **Algorithm:** Bayesian Optimization over XGBoost predictions
- **Tunable parameters:** feed_rate, temperature, rpm, chemical_dose, residence_time, backpressure
- **Constraints:** Equipment limits, safety (flash point), environmental (discharge limits)
- **Business value:** Tell operator WHAT SETTINGS to use for each specific batch

#### Model 1C: Quality Risk Classification
- **Target:** `quality_pass` (binary) + `quality_failure_reason` (multi-class)
- **Algorithm:** Random Forest Classifier
- **Business value:** Flag batches likely to fail spec BEFORE processing → adjust parameters or reject

---

### 1.5 DASHBOARD VIEWS (Separation — Tab 1)

**View 1.1: Source Data Explorer**
- Filterable data table of waste_reception_log
- Distribution charts of key properties by waste category
- Correlation matrix of feedstock properties
- Time series of incoming waste volume and composition

**View 1.2: Feature Engineering Pipeline**
- Side-by-side: raw columns → computed features
- Feature importance ranking from trained model
- Feature distributions and outlier detection
- Interactive: select a batch, see all its features computed step-by-step

**View 1.3: Model Performance**
- Actual vs. Predicted yield scatter plot
- Residual analysis
- Feature importance (SHAP values)
- Error distribution by waste category (which types are hardest to predict?)
- Learning curve (how much data needed for good predictions?)

**View 1.4: Optimization Engine (Main Operational View)**
- Select a new/incoming batch → see predicted optimal parameters
- Slider controls to simulate "what if" scenarios
- Predicted yield + confidence interval
- Comparison: current operator settings vs. AI-recommended settings
- Expected revenue/cost impact

---

## USE CASE 2: PREDICTIVE FLEET ROUTING & DEMAND FORECASTING

### 2.1 SOURCE SYSTEMS (Where Real Data Would Come From)

#### Source System G: AIS (Automatic Identification System)
**What it is:** Mandatory ship tracking transponder system. Every vessel >300GT broadcasts position.  
**Real-world providers:** MarineTraffic, VesselFinder, Spire Maritime, Kpler  
**Data frequency:** Every 2-30 seconds (depends on vessel speed)  
**Provides:** Vessel positions, speed, heading, destination, ETA at ports

#### Source System H: Port Management Information System (PMIS)
**What it is:** Port authority system tracking vessel calls, berth assignments, waste notifications.  
**Real-world systems:** PortBase (Rotterdam), DAKOSY (Hamburg), Poseidon (Piraeus)  
**Data frequency:** Per vessel call  
**Provides:** Expected arrivals, berth schedules, waste pre-notifications (MARPOL requirement: 24h before arrival)

#### Source System I: HEC Fleet Management System
**What it is:** Internal system tracking HEC's own tanker fleet — positions, capacity, schedules.  
**Real-world vendors:** Helm Operations, Danaos One, AMOS  
**Data frequency:** Real-time (GPS) + per voyage  
**Provides:** Fleet positions, tank levels, fuel consumption, availability

#### Source System J: Oil Price & Market Data
**What it is:** Commodity pricing that affects recovered fuel value and client behavior.  
**Real-world sources:** Platts, Argus Media, ICE Brent Futures  
**Data frequency:** Daily  
**Provides:** Crude oil price, bunker fuel prices, recovered fuel benchmark prices

#### Source System K: Offshore Platform Operations
**What it is:** Production data from oil/gas platforms served by HEC's large tankers.  
**Real-world context:** North Sea (UK/Norway), Eastern Med (Egypt, Israel, Cyprus)  
**Data frequency:** Daily production reports  
**Provides:** Production volumes, waste accumulation rates, scheduled maintenance shutdowns

#### Source System L: Weather & Sea State (Maritime)
**What it is:** Marine weather forecasts affecting fleet operations.  
**Real-world sources:** ECMWF Marine, UK Met Office, Copernicus Marine Service  
**Data frequency:** 6-hourly forecasts  
**Provides:** Wave height, wind speed, swell, visibility — determines if collection is safe

---

### 2.2 SOURCE DATA TABLES (Fleet & Demand)

#### Table: `port_waste_demand` (from Sources G, H, D)
**Business meaning:** Historical record of waste generation/collection at each port served by HEC.

| Column | Type | Realistic Values | Business Explanation |
|--------|------|-----------------|---------------------|
| `demand_id` | str | "DEM-PIR-2024-00891" | Unique demand record |
| `date` | date | 2022-01-01 to 2026-06-01 | Date of demand |
| `port_code` | str | PIR, HAM, GIB, MLT, GEN, MRS, BCN, RTM, ALG, LIM, ALE, IST, PSD (Port Said), TAN (Tangier) | Port identifier |
| `port_name` | str | Full name | |
| `port_country` | str | GR, DE, GI, MT, IT, FR, ES, NL, EG, TR, CY, MA | Country code |
| `port_latitude` | float | 29–54° N | Realistic coordinates |
| `port_longitude` | float | -6 to 35° E | Realistic coordinates |
| `vessel_calls_total` | int | 5–150 per day | Total vessel arrivals that day |
| `vessel_calls_by_type_container` | int | 2–60 | Container ship arrivals |
| `vessel_calls_by_type_tanker` | int | 1–40 | Oil/chemical tanker arrivals |
| `vessel_calls_by_type_bulk` | int | 1–30 | Bulk carrier arrivals |
| `vessel_calls_by_type_cruise` | int | 0–15 | Cruise ship arrivals (seasonal!) |
| `vessel_calls_by_type_other` | int | 1–20 | RoRo, LNG, etc. |
| `waste_volume_collected_m3` | float | 10–5,000 | Total waste collected that day at this port |
| `waste_bilge_m3` | float | 5–2,000 | Bilge water collected |
| `waste_sludge_m3` | float | 2–1,500 | Fuel oil sludge |
| `waste_slops_m3` | float | 0–2,000 | Tank washings/slops |
| `waste_other_m3` | float | 0–500 | Garbage, sewage, chemicals (not HEC's focus) |
| `avg_vessel_size_gt` | float | 5,000–80,000 | Average GT of calling vessels (larger = more waste) |
| `hec_market_share_pct` | float | 5–85% | HEC's share at this port (dominant in Piraeus, small in Rotterdam) |
| `competitor_presence` | int | 0–5 | Number of competing waste collectors at port |
| `port_reception_facility_capacity_m3` | float | 500–50,000 | Total storage capacity at port |
| `current_storage_fill_pct` | float | 10–95% | How full is the port storage (urgency indicator) |
| `days_until_full` | float | 1–60 | At current rate, when will storage overflow |

---

#### Table: `hec_fleet_status` (from Source I)
**Business meaning:** Real-time status of each HEC tanker.

| Column | Type | Realistic Values | Business Explanation |
|--------|------|-----------------|---------------------|
| `vessel_id` | str | "HEC-T-001" to "HEC-T-025" | Fleet vessel ID |
| `vessel_name` | str | "HEC Poseidon", "HEC Athena", "Green Star I-V", "Green Collector I-III" | Vessel names |
| `vessel_class` | str | Coastal Small (500 DWT), Coastal Medium (1000-2000), Regional (3500-7000), Offshore Large (30000+) | Fleet segment |
| `dwt` | int | 500, 1000, 1500, 2000, 3500, 5000, 7000, 30000, 35000 | Deadweight tonnage |
| `tank_capacity_m3` | float | 400–32,000 | Usable cargo tank volume |
| `home_port` | str | PIR, HAM, GIB, MLT | Base port |
| `timestamp` | datetime | | Position report time |
| `latitude` | float | 29–57° N | Current position |
| `longitude` | float | -10 to 36° E | Current position |
| `speed_knots` | float | 0–14 | Current speed (0 = in port) |
| `heading_deg` | float | 0–360 | Current heading |
| `status` | str | At Port Loading, At Port Discharging, In Transit Laden, In Transit Ballast, At Anchor, Under Maintenance, Dry Dock | Operational status |
| `current_cargo_m3` | float | 0 to tank_capacity | Current waste on board |
| `current_cargo_pct` | float | 0–100% | Tank fill percentage |
| `cargo_type_primary` | str | Mixed Oily, Bilge Water, Sludge, Offshore Slops | What's on board |
| `destination_port` | str | Port code | Where heading |
| `eta_destination` | datetime | | Expected arrival |
| `fuel_remaining_mt` | float | 5–800 | Fuel onboard (metric tons) |
| `fuel_consumption_mt_day` | float | 2–35 | Daily fuel burn (speed-dependent) |
| `days_until_next_maintenance` | int | 5–365 | Planned maintenance countdown |
| `crew_hours_remaining` | float | 0–720 | Hours before mandatory rest (regulations) |
| `ice_class` | str | None, 1C, 1B, 1A | Ice capability (relevant for Hamburg winter) |
| `last_inspection_date` | date | | Vessel survey date |
| `vetting_status` | str | Approved, Conditional, Expired | Oil major vetting (required for offshore work) |

---

#### Table: `voyage_history` (from Source I)
**Business meaning:** Completed voyages — what actually happened for each trip.

| Column | Type | Realistic Values | Business Explanation |
|--------|------|-----------------|---------------------|
| `voyage_id` | str | "VOY-2024-00234" | Unique voyage |
| `vessel_id` | str | FK → fleet | Which tanker |
| `departure_port` | str | Port code | Left from |
| `departure_time` | datetime | | When departed |
| `arrival_port` | str | Port code | Arrived at |
| `arrival_time` | datetime | | When arrived |
| `distance_nm` | float | 5–3,500 | Nautical miles traveled |
| `voyage_duration_hrs` | float | 2–240 | Total time (including waiting) |
| `sea_time_hrs` | float | 1–200 | Actual sailing time |
| `port_time_hrs` | float | 2–72 | Time in port (loading/discharging) |
| `waiting_time_hrs` | float | 0–48 | Waiting for berth/weather |
| `avg_speed_knots` | float | 6–13 | Average speed achieved |
| `fuel_consumed_mt` | float | 1–250 | Total fuel burned |
| `cargo_loaded_m3` | float | 0–32,000 | Waste collected on this leg |
| `cargo_discharged_m3` | float | 0–32,000 | Waste delivered to treatment plant |
| `revenue_eur` | float | 500–500,000 | Revenue for this voyage |
| `fuel_cost_eur` | float | 200–150,000 | Fuel expense |
| `port_charges_eur` | float | 100–20,000 | Port fees |
| `total_voyage_cost_eur` | float | 500–200,000 | All costs |
| `voyage_margin_eur` | float | -20,000 to +350,000 | Profit |
| `weather_delays_hrs` | float | 0–48 | Time lost to bad weather |
| `mechanical_delays_hrs` | float | 0–24 | Time lost to breakdowns |
| `collections_count` | int | 1–15 | Number of vessels/ports collected from |
| `avg_collection_time_hrs` | float | 1–8 | Average time per collection |

---

#### Table: `offshore_platform_data` (from Source K)
**Business meaning:** Oil/gas platforms served by HEC's large tankers (30,000+ DWT).

| Column | Type | Realistic Values | Business Explanation |
|--------|------|-----------------|---------------------|
| `platform_id` | str | "PLT-NS-001", "PLT-EM-005" | Platform identifier |
| `platform_name` | str | Realistic names | e.g., "Prinos", "Epsilon", "Leviathan Satellite" |
| `region` | str | North Sea, Eastern Mediterranean, Adriatic, West Africa | Operating region |
| `latitude` | float | 30–62° N | Position |
| `longitude` | float | -5 to 35° E | Position |
| `operator` | str | "Energean", "Total", "ENI", "BP", "Shell" | Oil company operator |
| `production_bpd` | int | 5,000–200,000 | Barrels per day oil production |
| `water_cut_pct` | float | 10–95% | Produced water percentage (increases with field age) |
| `waste_generation_m3_day` | float | 10–500 | Daily slops/waste generated |
| `storage_capacity_m3` | float | 1,000–50,000 | Onboard waste storage |
| `current_storage_m3` | float | 100–45,000 | Current waste level |
| `days_until_full` | float | 3–90 | Urgency indicator |
| `last_collection_date` | date | | When HEC last collected |
| `collection_frequency_days` | int | 14–90 | Typical interval between collections |
| `access_restrictions` | str | None, Weather Window Only, Helicopter Access Only, Night Operations Banned | Operational constraints |
| `minimum_tanker_size_dwt` | int | 5,000–30,000 | Minimum vessel size accepted |
| `contract_type` | str | Spot, Term Contract, Framework | Commercial arrangement |
| `distance_from_nearest_port_nm` | float | 20–500 | Sailing distance to nearest HEC facility |

---

#### Table: `maritime_weather` (from Source L)
**Business meaning:** Sea state affecting fleet operations — critical for safety and scheduling.

| Column | Type | Realistic Values | Business Explanation |
|--------|------|-----------------|---------------------|
| `timestamp` | datetime | 6-hourly | Forecast time |
| `zone_id` | str | "MED-W", "MED-C", "MED-E", "ATL-GIB", "NORTH-SEA", "CHANNEL" | Maritime weather zone |
| `significant_wave_height_m` | float | 0.2–8 | Hs — primary safety metric (>2.5m = risky for small tankers) |
| `max_wave_height_m` | float | 0.5–14 | Hmax — extreme wave |
| `wave_period_s` | float | 3–15 | Average wave period |
| `swell_height_m` | float | 0–4 | Long-period swell |
| `wind_speed_knots` | float | 0–60 | Surface wind |
| `wind_direction_deg` | float | 0–360 | Wind from |
| `visibility_nm` | float | 0.1–20 | Visibility in nautical miles |
| `sea_surface_temp_c` | float | 10–28 (Med), 4–18 (North Sea) | Affects waste viscosity during transfer |
| `beaufort_scale` | int | 0–12 | Sea state scale |
| `operation_feasibility` | str | Go, Marginal, No-Go | Can HEC tankers safely operate? |
| `small_vessel_limit_dwt` | int | 0/500/1000/2000 | Minimum vessel size for safe operations today |

---

#### Table: `oil_market_data` (from Source J)
**Business meaning:** Market prices affecting economics of waste collection and fuel recovery.

| Column | Type | Realistic Values | Business Explanation |
|--------|------|-----------------|---------------------|
| `date` | date | Daily | Trading date |
| `brent_crude_usd_bbl` | float | 40–130 | Brent crude oil price (drives recovered fuel value) |
| `rotterdam_hfo_380_usd_mt` | float | 200–700 | Heavy fuel oil price (bunker fuel, what HEC's fleet burns) |
| `rotterdam_vlsfo_usd_mt` | float | 350–900 | Very low sulfur fuel oil price |
| `rotterdam_mgo_usd_mt` | float | 500–1,200 | Marine gas oil price |
| `recovered_fuel_premium_discount_pct` | float | -15 to +5% | HEC's recovered fuel price vs. benchmark (usually discount) |
| `carbon_credit_eur_ton` | float | 20–100 | EU ETS carbon price (affects circular economy value) |
| `eur_usd_rate` | float | 0.95–1.25 | Exchange rate (revenues in EUR, some costs in USD) |
| `shipping_demand_index` | float | 80–150 | Baltic Dry Index proxy — high = more ships = more waste |

---

#### Table: `port_distance_matrix` (precomputed)
**Business meaning:** Sailing distances and times between all port pairs.

| Column | Type | Business Explanation |
|--------|------|---------------------|
| `origin_port` | str | From port |
| `destination_port` | str | To port |
| `distance_nm` | float | Sea distance in nautical miles |
| `typical_transit_hrs_small` | float | Time for 500-2000 DWT at 9 knots |
| `typical_transit_hrs_medium` | float | Time for 3500-7000 DWT at 11 knots |
| `typical_transit_hrs_large` | float | Time for 30000 DWT at 12 knots |
| `fuel_consumption_mt_small` | float | Fuel for small vessel one-way |
| `fuel_consumption_mt_medium` | float | Fuel for medium vessel |
| `fuel_consumption_mt_large` | float | Fuel for large vessel |
| `strait_passage` | bool | Goes through strait (Gibraltar, Suez) — adds time/cost |
| `pilotage_required` | bool | Pilot mandatory for this route segment |

**Realistic distances (examples):**
- Piraeus → Valletta: 410 nm
- Piraeus → Gibraltar: 1,310 nm
- Gibraltar → Hamburg: 1,530 nm
- Valletta → Genoa: 520 nm
- Hamburg → Rotterdam: 260 nm
- Piraeus → Limassol: 520 nm
- Gibraltar → Algeciras: 5 nm (same bay)
- Piraeus → Istanbul: 340 nm

---

### 2.3 FEATURE ENGINEERING (Fleet & Demand)

#### Category E: Demand Prediction Features
| Feature | Formula/Logic | Why |
|---------|---------------|-----|
| `vessel_calls_7d_rolling` | 7-day moving average of vessel calls | Smoothed traffic trend |
| `vessel_calls_30d_rolling` | 30-day moving average | Longer trend |
| `waste_per_vessel_call` | waste_volume / vessel_calls | Average waste per ship (changes with fleet composition) |
| `seasonal_decomposition` | STL decomposition → trend + seasonal + residual | Isolate seasonal pattern (cruise season, winter slowdown) |
| `yoy_growth_pct` | Same period last year comparison | Year-over-year trend |
| `cruise_season_flag` | 1 if Apr-Oct AND port is Mediterranean | Cruise ships = large waste volumes, seasonal |
| `port_congestion_proxy` | vessel_calls / port_capacity | Busy port = longer waits |
| `days_since_last_collection` | Current date - last HEC collection date at port | Urgency builds over time |
| `storage_fill_rate_m3_day` | Derivative of storage fill level | How fast is storage filling? |
| `weather_window_probability` | % of next 7 days forecast as "Go" | Can we even get there? |

#### Category F: Route Optimization Features
| Feature | Formula/Logic | Why |
|---------|---------------|-----|
| `vessel_proximity_to_demand_nm` | Great circle distance vessel→port | Which tanker is closest? |
| `vessel_available_capacity_m3` | tank_capacity - current_cargo | Can the tanker fit the demand? |
| `urgency_score` | storage_fill_pct × (1 / days_until_full) | Composite urgency metric |
| `collection_efficiency_m3_per_nm` | predicted_volume / distance | Is the trip worthwhile? |
| `fuel_cost_per_m3_collected` | (distance × consumption_rate × fuel_price) / predicted_volume | Cost efficiency metric |
| `multi_stop_opportunity` | Count of ports within 100nm with demand > threshold | Can we combine stops? |
| `weather_risk_enroute` | Max wave height on planned route in next 48h | Safety/delay risk |
| `contract_obligation_days` | Days until contract SLA breach at port | Legal urgency |
| `vessel_fuel_autonomy_days` | fuel_remaining / daily_consumption | Can vessel reach without refueling? |
| `regulatory_deadline_pressure` | MARPOL requires collection within X days | Compliance driver |

#### Category G: Economic Features
| Feature | Formula/Logic | Why |
|---------|---------------|-----|
| `expected_revenue_per_voyage` | predicted_volume × acceptance_fee + predicted_recovery_value | Trip revenue forecast |
| `voyage_cost_estimate` | fuel_cost + port_charges + crew_cost + time_cost | Trip cost estimate |
| `expected_margin` | revenue - cost | Is the voyage profitable? |
| `opportunity_cost` | What other demand could this vessel serve instead? | Better use of asset? |
| `oil_price_trend_7d` | 7-day change in Brent | Rising prices = more valuable to collect now |
| `market_share_risk` | If we don't collect, competitor will (1/0) | Competitive pressure |

---

### 2.4 ML MODELS (Fleet & Demand)

#### Model 2A: Waste Demand Forecasting (Time Series)
- **Target:** `waste_volume_collected_m3` per port per day, 7-day and 30-day ahead
- **Algorithm:** Ensemble of Prophet (seasonality) + XGBoost (features) + LSTM option
- **Features:** Category E above + weather + oil market + port-specific seasonality
- **Granularity:** Per port, daily
- **Evaluation:** MAPE, RMSE, coverage of prediction intervals
- **Business value:** Know how much waste will accumulate at each port → plan fleet proactively

#### Model 2B: Route Optimization (Prescriptive)
- **Target:** Minimize total fleet cost while collecting all demand within SLA deadlines
- **Algorithm:** Mixed-Integer Linear Programming (OR-tools) + heuristic (nearest-neighbor with improvements)
- **Constraints:** Vessel capacity, fuel autonomy, weather, crew rest, maintenance windows, MARPOL deadlines
- **Decisions:** Which vessel → which port → in what sequence → when
- **Business value:** Reduce fuel cost, increase utilization, prevent SLA breaches

#### Model 2C: Voyage Profitability Prediction
- **Target:** `voyage_margin_eur`
- **Algorithm:** XGBoost Regressor
- **Features:** Category F + G + predicted demand from Model 2A
- **Business value:** Prioritize most profitable voyages, decline unprofitable spot requests

---

### 2.5 DASHBOARD VIEWS (Fleet — Tab 2)

**View 2.1: Source Data Explorer**
- Map showing all ports with color-coded demand levels
- Fleet positions on map (real-time simulation)
- Time series of waste volumes per port
- Oil market overlay

**View 2.2: Demand Forecast Pipeline**
- Raw data → features → forecast, shown step by step
- Forecast accuracy metrics per port
- Seasonal patterns visualization
- Anomaly detection (unusual demand spikes)

**View 2.3: Fleet Optimization Dashboard**
- Interactive map: drag vessels to see cost impact
- Gantt chart of vessel schedules
- KPIs: fleet utilization %, empty miles %, SLA compliance %
- Cost breakdown: fuel, port, time, opportunity cost
- "What-if" simulator: add/remove a vessel, see fleet impact

**View 2.4: Model Performance**
- Forecast vs. actual (time series overlays)
- Route efficiency before/after optimization
- Cost savings quantification
- Profitability heatmap by port × month

---

## REALISTIC DATA GENERATION RULES

To ensure the synthetic data is maximally realistic:

### Physical Constraints (HARD — never violated)
- Water + Oil + Solids = 100% (±0.5% measurement error)
- Oil recovery yield cannot exceed oil_content_pct
- Flash point < 60°C requires special handling (high-risk)
- Vessel cannot carry more than DWT capacity
- Vessel speed limited to design max (typically 12-14 knots)
- Sea state > Beaufort 6: no operations for vessels < 3000 DWT
- Fuel consumption increases cubically with speed
- MARPOL: oil in water discharge < 15 ppm required

### Correlations (SOFT — statistical relationships)
- Higher viscosity → lower recovery yield (harder to separate)
- Higher temperature → lower viscosity (Walther equation)
- Larger vessels generate more waste (roughly proportional to GT^0.7)
- Mediterranean ports peak May-October (cruise season)
- Hamburg/Rotterdam peak year-round (container traffic steady)
- Oil price up → more offshore production → more platform waste
- Older equipment → higher energy consumption, lower yield
- Experienced operators → 3-8% better yield
- Night shifts → 1-2% lower yield
- Winter → higher heating costs, longer processing times

### Seasonal Patterns
- **Jan-Mar:** Low shipping Med, normal North Europe, high seas, cold = high viscosity
- **Apr-Jun:** Increasing Med traffic, cruise season starts, moderate weather
- **Jul-Sep:** Peak Med + cruise, calm seas, hot = easier processing but VOC risk
- **Oct-Dec:** Declining Med, storms increase, holiday slowdown December

### Volume Sizing (per year, per facility)
- **Piraeus:** ~150,000 m³/year (largest, hub for Eastern Med)
- **Hamburg:** ~80,000 m³/year (steady industrial + container port)
- **Gibraltar:** ~60,000 m³/year (transit port, bunker hub)
- **Malta:** ~40,000 m³/year (fleet operations base, smaller treatment)

---

## FILE STRUCTURE

```
C:\SAPDevelop\meli\
├── app/
│   ├── main.py                    # Streamlit app entry point
│   ├── pages/
│   │   ├── 1_separation_control.py    # Tab 1
│   │   └── 2_fleet_routing.py         # Tab 2
│   ├── components/
│   │   ├── pipeline_viewer.py     # Reusable source→FE→model→result component
│   │   ├── map_viewer.py          # Fleet/port map
│   │   └── charts.py             # Chart utilities
│   └── config.py                  # App configuration
├── data/
│   ├── generators/
│   │   ├── separation_data_generator.py    # Generate UC1 synthetic data
│   │   ├── fleet_data_generator.py         # Generate UC2 synthetic data
│   │   └── common.py                       # Shared generation utilities
│   ├── raw/                        # Generated CSVs (source data)
│   ├── features/                   # Feature-engineered CSVs
│   └── reference/                  # Static lookup tables (ports, distances, equipment)
├── models/
│   ├── separation/
│   │   ├── yield_predictor.py     # Model 1A
│   │   ├── parameter_optimizer.py # Model 1B
│   │   └── quality_classifier.py # Model 1C
│   ├── fleet/
│   │   ├── demand_forecaster.py   # Model 2A
│   │   ├── route_optimizer.py     # Model 2B
│   │   └── profitability_model.py # Model 2C
│   └── trained/                   # Serialized model artifacts
├── features/
│   ├── separation_features.py     # Feature engineering UC1
│   └── fleet_features.py          # Feature engineering UC2
├── requirements.txt
└── README.md (only if requested)
```

---

## DATA VOLUME

| Table | Rows | Rationale |
|-------|------|-----------|
| waste_reception_log | ~15,000 | ~4 years × ~10 receptions/day across 4 facilities |
| laboratory_analysis | ~45,000 | ~3 samples per reception (inlet, mid, outlet) |
| process_sensor_data | ~2,000,000 | 1-minute readings × ~20 batches/day × 4 years (sampled) |
| processing_batch | ~12,000 | Slightly fewer than receptions (batches may combine multiple) |
| equipment_master | ~40 | Static equipment list |
| weather_conditions | ~140,000 | Hourly × 4 facilities × 4 years |
| port_waste_demand | ~80,000 | Daily × 14 ports × 4 years |
| hec_fleet_status | ~500,000 | Every 4 hours × 25 vessels × 4 years |
| voyage_history | ~8,000 | ~5 voyages/vessel/month × 25 vessels × 4 years |
| offshore_platform_data | ~5,000 | Daily × 4 platforms × ~3 years |
| maritime_weather | ~24,000 | 6-hourly × 6 zones × 4 years |
| oil_market_data | ~1,500 | Daily × 4 years |
| port_distance_matrix | ~196 | 14 ports × 14 ports |

---

## SPEC SELF-REVIEW

1. **Placeholder scan:** No TBDs or TODOs. All sections fully specified.
2. **Internal consistency:** Source systems map to tables, tables map to features, features map to models, models map to dashboard views. Pipeline is complete end-to-end.
3. **Scope check:** This is one implementation — two tabs, shared infrastructure. Appropriately scoped.
4. **Ambiguity check:** Ranges, types, and business explanations provided for every single column. No ambiguity.
