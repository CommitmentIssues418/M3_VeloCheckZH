"""
velocheck_pipeline.py

Data pipeline for the M3_VeloCheckZH project.

It reproduces the main preprocessing steps from the Colab notebook:
- load holidays, create holiday_status
- load weather data
- load & aggregate bike counting data
- merge everything into a modeling table

Run from the project root like:
    python -m src.velocheck_pipeline
or
    python src/velocheck_pipeline.py
"""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# === 1) PATH SETUP (ADAPT FILENAMES IF NEEDED) ==============================

# Project root = two levels above this file: .../M3_VeloCheckZH
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data folder
DATA_DIR = PROJECT_ROOT / "data"

# Optional subfolders (you can change these to your structure)
WEATHER_FILE = DATA_DIR / "weather" / "ogd-smn_reh_d_historical.csv"
HOLIDAYS_FILE = DATA_DIR / "holidays" / "schulferien.csv"
COUNTS_2022_FILE = DATA_DIR / "counts" / "2022_verkehrszaehlungen_werte_fussgaenger_velo.csv"
COUNTS_2023_FILE = DATA_DIR / "counts" / "2023_verkehrszaehlungen_werte_fussgaenger_velo.csv"
COUNTS_2024_FILE = DATA_DIR / "counts" / "2024_verkehrszaehlungen_werte_fussgaenger_velo.csv"


def load_holiday_status():
    """Load school holidays and build a daily is_holiday table for 2022–2024."""
    holidays = pd.read_csv(HOLIDAYS_FILE)

    if "created_date" in holidays.columns:
        holidays = holidays.drop(columns=["created_date"])

    # Parse as UTC, then make tz-naive
    holidays["Startdatum"] = pd.to_datetime(holidays["start_date"], utc=True).dt.tz_localize(None)
    holidays["Enddatum"] = pd.to_datetime(holidays["end_date"], utc=True).dt.tz_localize(None)
    holidays = holidays.drop(columns=["summary", "start_date", "end_date"])

    # Create daily date range as tz-naive (default)
    all_dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")
    holiday_status = pd.DataFrame({"Datum": all_dates})
    holiday_status["is_holiday"] = False
    holiday_status["Zeitstempel"] = holiday_status["Datum"]

    for _, row in holidays.iterrows():
        start_date = row["Startdatum"]
        end_date = row["Enddatum"] - pd.Timedelta(days=1)
        mask = (holiday_status["Datum"] >= start_date) & (
            holiday_status["Datum"] <= end_date
        )
        holiday_status.loc[mask, "is_holiday"] = True

    return holiday_status




def load_weather():
    """Load weather CSV and return filtered dataframe for 2022–2024."""
    weather_raw = pd.read_csv(WEATHER_FILE, sep=";")

    weather_all = weather_raw[
        ["reference_timestamp", "rka150d0", "tre200d0", "tre200dn", "tre200dx"]
    ].copy()

    weather_all = weather_all.rename(
        columns={
            "reference_timestamp": "Zeitstempel",
            "rka150d0": "Niederschlag",
            "tre200d0": "Temperatur Durchschnitt",
            "tre200dn": "Temperatur min",
            "tre200dx": "Temperatur max",
        }
    )

    weather_all["Datum"] = pd.to_datetime(
        weather_all["Zeitstempel"], format="%d.%m.%Y %H:%M"
    )

    for col in [
        "Niederschlag",
        "Temperatur Durchschnitt",
        "Temperatur min",
        "Temperatur max",
    ]:
        weather_all[col] = pd.to_numeric(weather_all[col], errors="coerce")

    weather = weather_all[
        (weather_all["Datum"].dt.year >= 2022) & (weather_all["Datum"].dt.year <= 2024)
    ].reset_index(drop=True)

    return weather


def load_daily_counts():
    """Load bike counting data, aggregate to daily level, and add features."""
    df_list = []
    for f in [COUNTS_2022_FILE, COUNTS_2023_FILE, COUNTS_2024_FILE]:
        if f.exists():
            df_list.append(pd.read_csv(f))
        else:
            print(f"Warning: file not found: {f}")

    if not df_list:
        raise FileNotFoundError("No counting data CSVs were found in data/counts/")

    counts_raw = pd.concat(df_list, ignore_index=True)

    counts_all = counts_raw[
        ["FK_STANDORT", "DATUM", "VELO_IN", "VELO_OUT", "OST", "NORD"]
    ].copy()

    counts_all = counts_all.rename(
        columns={
            "FK_STANDORT": "Standort_ID",
            "DATUM": "Zeitstempel",
            "VELO_IN": "Velo_In",
            "VELO_OUT": "Velo_Out",
            "OST": "Koord_Ost",
            "NORD": "Koord_Nord",
        }
    )

    counts_all["Zeitstempel"] = pd.to_datetime(counts_all["Zeitstempel"])
    for col in ["Velo_In", "Velo_Out", "Koord_Ost", "Koord_Nord"]:
        counts_all[col] = pd.to_numeric(counts_all[col], errors="coerce")

    counts_all["Datum"] = counts_all["Zeitstempel"].dt.normalize()

    daily_counts = (
        counts_all.groupby(["Standort_ID", "Datum", "Koord_Ost", "Koord_Nord"])[
            ["Velo_In", "Velo_Out"]
        ]
        .sum()
        .reset_index()
    )

    daily_counts["Velo"] = daily_counts["Velo_In"] + daily_counts["Velo_Out"]
    daily_counts["year"] = daily_counts["Datum"].dt.year
    daily_counts["day_of_year"] = daily_counts["Datum"].dt.dayofyear

    daily_counts = daily_counts[daily_counts["Velo"] != 0]

    return daily_counts


def create_plot(daily_counts: pd.DataFrame):
    """Simple line plot of total Velo counts over time."""
    plt.figure(figsize=(15, 7))
    sns.lineplot(data=daily_counts, x="Datum", y="Velo", legend=False, linewidth=0.8)
    plt.title("Tägliche Velozählungen pro Zählstelle über das Jahr")
    plt.xlabel("Datum")
    plt.ylabel("Anzahl Velos")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def build_model_table(daily_counts, weather, holiday_status):
    """Merge all sources and return final data_for_model."""
    merged = pd.merge(daily_counts, weather, on="Datum", how="outer")

    merged["weekday"] = merged["Datum"].dt.weekday
    merged["weekend"] = merged["weekday"] >= 5

    merged = pd.merge(
        merged, holiday_status[["Datum", "is_holiday"]], on="Datum", how="left"
    )

    cols_to_drop = [
        "Zeitstempel",      # from weather
        "Velo_In",
        "Velo_Out",
        "Temperatur min",
        "Temperatur max",
    ]
    cols_to_drop = [c for c in cols_to_drop if c in merged.columns]
    data_for_model = merged.drop(columns=cols_to_drop)

    return data_for_model


def main():
    print("=== M3_VeloCheckZH pipeline ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir:     {DATA_DIR}")

    # 1) Holiday status
    print("\n[1] Loading holidays...")
    holiday_status = load_holiday_status()
    print(holiday_status.head())

    # 2) Weather
    print("\n[2] Loading weather...")
    weather = load_weather()
    print(weather.tail())

    # 3) Counting data
    print("\n[3] Loading daily counts...")
    daily_counts = load_daily_counts()
    print(daily_counts.tail())

    # 4) Plot
    print("\n[4] Creating plot...")
    create_plot(daily_counts)

    # 5) Model table
    print("\n[5] Building modeling table...")
    data_for_model = build_model_table(daily_counts, weather, holiday_status)
    print(data_for_model.head(20))

    # Optionally save:
    out_path = DATA_DIR / "processed" / "data_for_model.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_for_model.to_csv(out_path, index=False)
    print(f"\nSaved data_for_model to: {out_path}")


if __name__ == "__main__":
    main()
