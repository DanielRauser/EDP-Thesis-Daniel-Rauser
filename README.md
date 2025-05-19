# EDP Thesis – Daniel Rauser

A reproducible end-to-end machine learning pipeline and exploratory analysis for analyzing & forecasting residual load, developed as part of Daniel Rauser’s EDP Thesis.

---

## 🚀 Project Overview

This repository implements a modular ML pipeline—powered by MLflow—for:
- **Data preparation** (climate data, market prices, plant characteristics, consumption history)  
- **Model training** (LightGBM or neural networks with hyperparameter optimization)  
- **Application** (generating out-of-sample forecasts across multiple assets and time horizons)  
- **Evaluation** (metrics, backtests, and scenario analysis)  

In addition, a suite of Jupyter Notebooks provides exploratory data analysis, time-series modeling (e.g. SARIMA), and interactive mapping for Portuguese power plants.

---

## 🔍 Repository Structure

│
├── Analysis/                    ← Jupyter Notebooks & visualizations
│   ├── Analysis_Consumption.ipynb
│   ├── Forecast_EDP.ipynb
│   ├── portugal_plants_map.html
│   └── …
│
├── Data/                        ← Raw and processed data files
│   ├── Day-ahead Market Prices_20230101_20250331.csv
│   ├── Breakdown of Production_*.xlsx
│   ├── ML_Consumption_Data.csv
│   └── …
│
├── Functions/                   ← Reusable modules for each pipeline step
│   ├── Dataprep/                ← climate, plant, solar, redes dataprep
│   ├── Apply/                   ← result-assembly & post-processing
│   ├── Evaluation/              ← performance-metric helpers
│   ├── Models/                  ← forecasting & simulation classes
│   ├── Optimizers/              ← hyperparameter search algorithms
│   └── Preprocessing/           ← feature engineering utilities
│
├── config/                      ← YAML configuration for pipeline control
│   └── config.yaml
│
├── pipeline/                    ← Task definitions (for MLflow runs)
│   ├── dataprep.py
│   ├── train.py
│   ├── apply.py
│   └── evaluate.py
│
├── main.py                      ← Entry point to orchestrate full pipeline
├── .gitattributes
└── .DS_Store


---

## ⚙️ Prerequisites

- **Python 3.12+**  
- **MLflow** (≥ 2.0)  
- **PyYAML**, **numpy**, **pandas**, **scikit-learn**, **lightgbm**  
- **Jupyter Lab / Notebook** (for Analysis/)  
- Other libs as needed by notebooks: `matplotlib`, `folium`, `geopandas`, etc.

---

## 📥 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/DanielRauser/EDP-Thesis-Daniel-Rauser.git
   cd EDP-Thesis-Daniel-Rauser

2. **Create & activate a virtual environment**

   python -m venv .venv
  source .venv/bin/activate      # Linux/macOS
  .venv\Scripts\activate         # Windows

3. **Install core dependencies**

   pip install requirements.txt

## 🔧 Configuration

Edit config/config.yaml to point to your data directories, choose which steps to run (dataprep, train, apply, evaluate), and set hyperparameters or experiment names.

**Example:**

experiment:
  name: "EDP_Forecasting"
dataprep:
  run: true
  input_path: "Data/"
  output_path: "Data/processed/"
train:
  run: true
  model: "lightgbm"
  folds: 5
apply:
  run: true
evaluate:
  run: true

## ▶️ Usage

Run the full pipeline (from data prep through evaluation) with:

python main.py \
  --config config/config.yaml \
  --experiment "EDP_Thesis_Experiment"

You can also skip steps via the config flags (use_existing, run: false) or supply an existing MLflow run ID.

## 📊 Analysis Notebooks

All exploratory analyses are in the Analysis folder

## 📄 License

**CC** Creative Commons
