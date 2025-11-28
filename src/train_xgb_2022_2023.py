"""
Train XGBoost model on 2022 + 2023 data and evaluate on 2024.

This script uses the engineered feature table:
    data/processed/data_for_model_engineered.csv
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_engineered_data(filename="data_for_model_engineered.csv"):
    """Load engineered feature table using a robust absolute path."""
    # Path zum Projekt-Root herausfinden
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # finaler Pfad
    path = os.path.join(project_root, "data", "processed", filename)

    print("Loading:", path)

    df = pd.read_csv(path)
    df["Datum"] = pd.to_datetime(df["Datum"])
    df = df.sort_values("Datum").reset_index(drop=True)

    target_col = "Velo"

    exclude = [
    "Velo",
    "Velo_log",       # nur falls vorhanden
    "Datum",          # rohes Datum nicht direkt ins Modell
    "Standort_ID",
    "Koord_Ost",
    "Koord_Nord",
    "year",
    "week",
    "quarter",
    # "day_of_year",
    "year_length",
    ]

    features = [c for c in df.columns if c not in exclude]

    X = df[features]
    y = df[target_col]
    dates = df["Datum"]

    return df, X, y, dates, features


def train_test_split_time(X, y, dates, cutoff="2024-01-01"):
    """Split data into train (bis 2023) und test (2024) anhand eines Datums-Cutoffs."""
    cutoff_date = pd.to_datetime(cutoff)

    train_idx = dates < cutoff_date
    test_idx = dates >= cutoff_date

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


def build_xgb_model(random_state=42):
    """Create XGBoost regressor with the hyperparameters used in the notebook."""
    model = XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    return model


# --------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------

def main():
    print("=== Train XGBoost on 2022–2023 and evaluate on 2024 ===")

    # 1. Load data
    df, X, y, dates, features = load_engineered_data()
    print(f"[1] Loaded engineered data: {df.shape[0]} rows, {len(features)} features.")

    # 2. Time-based split
    X_train, X_test, y_train, y_test = train_test_split_time(X, y, dates, cutoff="2024-01-01")
    print(f"[2] Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 3. Train model
    print("[3] Training XGBoost model...")
    model = build_xgb_model()
    model.fit(X_train, y_train)

    # 4. Evaluate on 2024
    print("[4] Evaluating on 2024...")
    y_pred = model.predict(X_test)

    rmse_2024 = rmse(y_test, y_pred)
    mae_2024 = mean_absolute_error(y_test, y_pred)

    print(f"    RMSE 2024: {rmse_2024:.3f}")
    print(f"    MAE 2024 : {mae_2024:.3f}")

    # 5. Save model
    # 5. Save model (absolute project-root path for safety)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_out = os.path.join(models_dir, "xgb_model_2022_2023.json")
    model.save_model(model_out)

    print(f"[5] Saved XGBoost model to: {model_out}")



if __name__ == "__main__":
    main()
