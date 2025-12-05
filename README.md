# CAS ADS Module 3: Modelling average cycling counts from short-term counting
CAS Applied Data Science – Module 3  
**Group Project by: Tamara, Joel, Charis**

This repository contains our complete end-to-end data science project for Module 3 of the CAS Applied Data Science program (University of Bern).  
The goal of the project is to **model and predict daily bicycle traffic at Zurich counting stations** using:
- historical bicycle counts  
- weather variables  
- calendar structure (weekday, weekend, holidays)  
- spatial information (station coordinates, regions)  
- engineered temporal and rolling features  
- machine learning models, including XGBoost  

The project is designed to be **fully reproducible**, with a clean separation of:
- **data pipeline** (`src/`)
- **feature engineering** (`src/`)
- **notebooks for EDA, model exploration & reporting** (`notebooks/`)
- **final model training script** (`src/train_xgb_2022_2023.py`)

---

# 🧠 When to Use Notebooks vs. Python Scripts

### ✔ **Jupyter Notebooks (`notebooks/`)**  
Used for:
- exploratory data analysis (EDA)  
- visualization  
- testing feature ideas  
- comparing models  
- SHAP explainability  
- reporting & slide preparation  
- time-based cross validation  

Notebooks **tell the story** of the project and show how results were derived.

---

### ✔ **Python Scripts (`src/`)**  
Used for reproducible, automated tasks:
- loading & preprocessing data  
- merging weather/holiday counts  
- feature engineering (rolling windows, lags, clustering, etc.)  
- final ML training outside notebooks  
- generating consistent results for production-like use  

Scripts **generate the actual data artifacts** used by the model.

---

# 🔧 Environment Setup

## 1. Clone the repository

```
git clone git@github.com:CommitmentIssues418/M3_VeloCheckZH.git
cd M3_VeloCheckZH
```

## 2. Create a virtual environment
Windows (PowerShell / Git Bash):
```python -m venv .venv
source .venv/Scripts/activate
```

macOS / Linux:
```python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies
```
pip install -r requirements.txt
```

The following main libraries are used:
- pandas, numpy, scipy
- scikit-learn
- XGBoost
- matplotlib, seaborn
- geopandas (for spatial EDA)
- statsmodels
- shap
- jupyter, notebook, ipykernel

# 🚴 Data Processing Pipeline (Reproducible Workflow)
## 1️⃣ Prepare the base feature table
Runs the full data cleaning pipeline, including merging raw velo, weather, and holiday data.
`python src/velocheck_pipeline.py`

This produces:
`data/processed/data_for_model.csv`

## 2️⃣ Feature engineering

Adds:
-month, week, quarter
- cyclic season encoding (sin_day, cos_day)
- temperature transformations
- rain indicators (is_rain, heavy_rain)
- rolling and lag features (velo_roll3, velo_roll7, velo_lag1, velo_lag7)
- station average traffic
- spatial KMeans clusters

Run:
`python src/feature_engineering.py`

Output:
`data/processed/data_for_model_engineered.csv`

## 3️⃣ Final model training (2022–2023 → 2024)
`src/train_xgb_2022_2023.py`
This script:
- loads engineered data
- trains an XGBoost model on years 2022–2023
- evaluates performance on unseen 2024
- prints RMSE & MAE
- saves final model to: `models/xgb_model_2022_2023.json`

# 📓 Notebook Overview
## 01_eda.ipynb
Exploratory analysis:
- global trends
- station patterns
- weekday/month effects
- weather relationships
- spatial plots

## 02_feature_engineering.ipynb
- Interactive testing and evaluation of engineered features.

## 03_model_training.ipynb
- Comparison of baseline, linear regression, random forest, and XGBoost.

## 04_model_explainability.ipynb
- SHAP summary, dependence, interaction effects.

## 05_error_analysis.ipynb
Detailed performance breakdown:
- by station
- by cluster
- by month
- insights about difficult/stable locations

## 06_time_cross_validation.ipynb
Train on 2022–2023 → Test on 2024
Includes:
- true vs predicted plots
- rolling averages
- error interpretation
