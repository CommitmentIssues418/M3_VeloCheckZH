"""
Feature Engineering for M3_VeloCheckZH.

This script:
- loads the base model table (daily velo counts + weather + holidays)
- adds time-based features
- adds weather-derived features
- adds rolling and lag features per station
- adds station-level traffic averages
- performs KMeans clustering on coordinates and adds location_cluster

Input:
    data/processed/data_for_model.csv

Output:
    data/processed/data_for_model_engineered.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# --------------------------------------------------------------------
# Helpers for robust paths
# --------------------------------------------------------------------

def get_project_root():
    """Return absolute path to project root (one level above src)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_base_table(filename="data_for_model.csv"):
    """Load base table from data/processed."""
    root = get_project_root()
    path = os.path.join(root, "data", "processed", filename)
    print(f"[LOAD] Loading base table from: {path}")
    df = pd.read_csv(path)
    df["Datum"] = pd.to_datetime(df["Datum"])
    df = df.sort_values(["Standort_ID", "Datum"]).reset_index(drop=True)
    return df


def save_engineered_table(df, filename="data_for_model_engineered.csv"):
    """Save engineered table to data/processed."""
    root = get_project_root()
    out_path = os.path.join(root, "data", "processed", filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[SAVE] Writing engineered table to: {out_path}")
    df.to_csv(out_path, index=False)


# --------------------------------------------------------------------
# Feature blocks
# --------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add richer time-based features: month, week, quarter, year_length, sin/cos of day_of_year."""
    print("[FE] Adding time features...")

    # Sicherstellen, dass Datum existiert
    if "Datum" not in df.columns:
        raise KeyError("Column 'Datum' not found in DataFrame.")

    # Basis: month, week, quarter
    df["month"] = df["Datum"].dt.month
    df["week"] = df["Datum"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["Datum"].dt.quarter

    # year_length (Schaltjahr vs nicht)
    df["year_length"] = np.where(df["Datum"].dt.is_leap_year, 366, 365)

    # Falls day_of_year noch nicht existiert, erstellen
    if "day_of_year" not in df.columns:
        df["day_of_year"] = df["Datum"].dt.dayofyear

    # Zyklische Kodierung
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / df["year_length"])
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / df["year_length"])

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived weather features like temp_feels, temp_sq, is_rain, heavy_rain."""
    print("[FE] Adding weather features...")

    # Temperatur Durchschnitt wird vorausgesetzt
    if "Temperatur Durchschnitt" not in df.columns:
        raise KeyError("Column 'Temperatur Durchschnitt' not found in DataFrame.")

    # Einfaches "Feels like" = hier identisch, aber separater Kanal für Modell
    df["temp_feels"] = df["Temperatur Durchschnitt"]

    # Quadratischer Term für Nichtlinearität
    df["temp_sq"] = df["Temperatur Durchschnitt"] ** 2

    # Niederschlag: Regenindikatoren
    if "Niederschlag" not in df.columns:
        raise KeyError("Column 'Niederschlag' not found in DataFrame.")

    df["is_rain"] = (df["Niederschlag"] > 0).astype(int)
    # Threshold kannst du bei Bedarf anpassen (z. B. 10mm)
    df["heavy_rain"] = (df["Niederschlag"] >= 10).astype(int)

    return df


def add_rolling_lag_features(
    df: pd.DataFrame,
    group_cols=("Standort_ID",),
    target_col="Velo",
) -> pd.DataFrame:
    """
    Add rolling and lag features per station:
    - velo_roll3, velo_roll7
    - velo_lag1, velo_lag7
    Assumes df is sorted by group_cols + Datum.
    """
    print("[FE] Adding rolling and lag features...")

    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in DataFrame.")

    # Defensive sort
    df = df.sort_values(list(group_cols) + ["Datum"]).reset_index(drop=True)

    # Rolling and lag per group
    df["velo_roll3"] = (
        df.groupby(list(group_cols))[target_col]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )
    df["velo_roll7"] = (
        df.groupby(list(group_cols))[target_col]
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    )

    df["velo_lag1"] = (
        df.groupby(list(group_cols))[target_col]
        .shift(1)
    )
    df["velo_lag7"] = (
        df.groupby(list(group_cols))[target_col]
        .shift(7)
    )

    # Erste Tage ohne Lag füllen wir optional mit roll3/roll7 oder NaN belassen
    # Hier: NaN belassen, Model kann damit umgehen (oder später im Notebook impute)
    return df


def add_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add station-level average traffic (station_avg_traffic)."""
    print("[FE] Adding station-level features...")

    if "Standort_ID" not in df.columns:
        raise KeyError("Column 'Standort_ID' not found in DataFrame.")
    if "Velo" not in df.columns:
        raise KeyError("Column 'Velo' not found in DataFrame.")

    station_avg = df.groupby("Standort_ID")["Velo"].mean()
    df["station_avg_traffic"] = df["Standort_ID"].map(station_avg)

    return df


def add_location_clusters(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    Perform KMeans clustering on station coordinates and add 'location_cluster'.
    Uses mean Koord_Ost/Koord_Nord per Standort_ID.
    """
    print(f"[FE] Adding location clusters with KMeans (k={n_clusters})...")

    for col in ("Koord_Ost", "Koord_Nord"):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    # Station-level coordinate table
    station_coords = (
        df.groupby("Standort_ID")[["Koord_Ost", "Koord_Nord"]]
        .mean()
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(station_coords)

    station_cluster = dict(zip(station_coords.index, kmeans.labels_))
    df["location_cluster"] = df["Standort_ID"].map(station_cluster)

    return df


# --------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------

def main():
    print("=== Feature Engineering for M3_VeloCheckZH ===")

    # 1. Load base table
    df = load_base_table("data_for_model.csv")
    print(f"[INFO] Base table shape: {df.shape}")

    # 2. Time features
    df = add_time_features(df)

    # 3. Weather features
    df = add_weather_features(df)

    # 4. Rolling and lag features
    df = add_rolling_lag_features(df, group_cols=("Standort_ID",), target_col="Velo")

    # 5. Station-level features
    df = add_station_features(df)

    # 6. Location clusters
    df = add_location_clusters(df, n_clusters=3, random_state=42)

    print(f"[INFO] Final engineered table shape: {df.shape}")

    # 7. Save
    save_engineered_table(df, "data_for_model_engineered.csv")

    print("=== Done. ===")


if __name__ == "__main__":
    main()
