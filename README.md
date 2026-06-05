# HEC AI Platform — Operations Intelligence

AI-powered operations platform for **Hellenic Environmental Center (HEC)**, optimizing petroleum waste separation processes and fleet routing through machine learning.

## Use Cases

### 1. AI-Optimized Separation Process Control
ML on real-time sensor data to dynamically adjust 3-phase petroleum waste separation parameters, maximising fuel recovery yield across 4 facilities (Piraeus, Hamburg, Gibraltar, Malta).

### 2. Predictive Fleet Routing & Demand Forecasting
ML demand prediction for waste generation per port/rig + route optimization for a 25-vessel collection tanker fleet operating in the Mediterranean and Northern Europe.

## Architecture

```
Source Data → Analytics → Feature Engineering → ML Model → Optimization Results
```

Every dashboard view follows this 5-step pipeline with full transparency.

## Quick Start

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Build the Data Pipeline

```bash
# 1. Generate synthetic datasets (756K+ rows across 11 tables)
python scripts/generate_data.py

# 2. Engineer features (separation: 12K×48, fleet: 22K×47, routes: 35K×12)
python scripts/engineer_features.py

# 3. Train ML models (XGBoost, Random Forest, scipy optimizers)
python scripts/train_models.py
```

Total pipeline runtime: ~2 minutes.

### Launch the Dashboard

```bash
streamlit run app/main.py
```

Open http://localhost:8501

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI/Dashboard | Streamlit (multi-page) |
| Charts | Plotly (interactive) |
| Maps | Folium + streamlit-folium |
| ML Models | XGBoost, scikit-learn, scipy |
| Data | pandas, numpy |
| Pipeline | Custom Python scripts |

## Models

| Model | Task | Performance |
|-------|------|-------------|
| Yield Predictor | Predict oil recovery yield | R²=0.775 |
| Quality Classifier | Classify batch quality | Acc=85.6% |
| Demand Forecaster | Predict port waste demand | R²=0.999 |
| Profitability Model | Predict voyage profitability | R²=0.949 |
| Parameter Optimizer | Optimize separation settings | scipy minimize |
| Route Optimizer | Optimize fleet routing | Greedy + scoring |

## Project Structure

```
meli/
├── app/
│   ├── main.py                    # Landing page
│   ├── config.py                  # Central configuration
│   ├── components/                # Shared UI components
│   │   ├── theme.py              # HEC corporate CSS + helpers
│   │   ├── charts.py             # 15+ interactive chart functions
│   │   ├── pipeline_viewer.py    # Pipeline step visualizer
│   │   └── map_viewer.py         # Folium map component
│   └── pages/
│       ├── 1_separation_control.py   # Use Case 1 (5 tabs)
│       └── 2_fleet_routing.py        # Use Case 2 (5 tabs)
├── data/
│   ├── generators/               # Synthetic data generators
│   ├── raw/                      # Generated raw datasets
│   ├── features/                 # Engineered feature sets
│   └── reference/                # Static reference data (ports, fleet)
├── features/                     # Feature engineering modules
├── models/
│   ├── separation/               # Separation model training
│   ├── fleet/                    # Fleet model training
│   └── trained/                  # Trained model artifacts
├── scripts/                      # Pipeline orchestration scripts
├── requirements.txt
└── README.md
```

## Deployment

This app is configured for **Streamlit Community Cloud** deployment:

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `app/main.py`
5. Deploy

The app will automatically run the data pipeline on first launch (see `.streamlit/config.toml`).

## Financial Impact

- **Separation**: Each 1% yield improvement = €1,500/batch = €18M/year
- **Fleet**: Route optimization reduces fuel costs by 8-15% per voyage
- **Combined AI opportunity**: €20M+ annual optimization potential

## License

Proprietary — Hellenic Environmental Center
