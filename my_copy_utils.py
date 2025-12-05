import os
import pandas as pd
from meteostat import Point, Daily
from datetime import datetime
import sqlite3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# --- Configuration ---
CITY_COORDS = {
    "gqeberha": (-33.9613, 25.6122),
    "port elizabeth": (-33.9613, 25.6122),
    "east london": (-33.0153, 27.9116),
    "mthatha": (-31.5893, 28.7840),
    "king williams town": (-32.8764, 27.4438),
    "bhisho": (-32.8490, 27.3670)
}

INDICATORS_INFO = {
    "tavg": "Average Temperature (°C)",
    "prcp": "Precipitation (mm)",
    "high_humidity": "Relative Humidity (%) (simulated)",
    "flood_risk": "Flood Risk Index (simulated)",
    "thunderstorm_days": "Thunderstorm Days Count (simulated)"
}

# --- User Inputs ---
print("Available cities in Eastern Cape:")
for i, city in enumerate(CITY_COORDS.keys(), start=1):
    print(f"{i}. {city.title()}")

city_choice = int(input("Choose a city by number: "))
city_name = list(CITY_COORDS.keys())[city_choice - 1]
lat, lon = CITY_COORDS[city_name]
print(f"Selected city: {city_name.title()} with coordinates: {lat}, {lon}")

print("\nAvailable indicators:")
for i, (col, desc) in enumerate(INDICATORS_INFO.items(), start=1):
    print(f"{i}. {desc} = {col}")

chosen = input("Enter the numbers of indicators to use (comma separated): ")
numbers = [int(x.strip()) for x in chosen.split(",")]
selected_indicators = [list(INDICATORS_INFO.keys())[n - 1] for n in numbers]
print("Selected indicators:", selected_indicators)

print("\nEnter the date range (YYYY-MM-DD):")
start_input = input("Start date: ")
end_input = input("End date: ")
start_date = datetime.strptime(start_input, "%Y-%m-%d")
end_date = datetime.strptime(end_input, "%Y-%m-%d")

# --- Fetch / Simulate Data ---
location = Point(lat, lon)
df = Daily(location, start_date, end_date).fetch()

# Keep only fetched indicators that exist in data
df = df[[col for col in selected_indicators if col in df.columns]]

# Simulate custom indicators, if selected
if "high_humidity" in selected_indicators:
    df["high_humidity"] = 70 + np.random.rand(len(df)) * 25  # ~70–95%
if "flood_risk" in selected_indicators:
    df["flood_risk"] = np.random.rand(len(df)) * 100
if "thunderstorm_days" in selected_indicators:
    df["thunderstorm_days"] = np.random.randint(0, 2, len(df))

# Filter to exactly selected indicators
df = df[[col for col in selected_indicators if col in df.columns]]

# --- Save Data ---
os.makedirs("weather_reports", exist_ok=True)
base_name = f"weather_{city_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

save_ops = {
    "csv": df.to_csv,
    "xlsx": df.to_excel,
    "json": lambda path: df.to_json(path, orient="records", date_format="iso"),
    "xml": lambda path: df.to_xml(path, root_name="weather", row_name="day"),
    "sql": lambda path: df.to_sql("weather", sqlite3.connect(path), if_exists="replace", index=True)
}

for fmt, op in save_ops.items():
    path = os.path.join("weather_reports", f"{base_name}.{fmt if fmt != 'sql' else 'db'}")
    op(path)
    print("Saved:", path)

# --- Load Data Option ---
print("\nChoose format to load/use:")
options = ["cloud"] + list(save_ops.keys())
for i, opt in enumerate(options, start=1):
    print(f"{i}. {opt}")

choice = int(input("Enter your choice number: "))
current = options[choice - 1]

load_ops = {
    "cloud": lambda _: df,
    "csv": lambda p: pd.read_csv(p, index_col=0, parse_dates=True),
    "xlsx": lambda p: pd.read_excel(p, index_col=0, parse_dates=True),
    "json": lambda p: pd.read_json(p).set_index("time"),
    "xml": lambda p: pd.read_xml(p).set_index("time"),
    "sql": lambda p: pd.read_sql("SELECT * FROM weather", sqlite3.connect(p), parse_dates=["time"], index_col="time")
}

file_path = None if current == "cloud" else os.path.join("weather_reports", f"{base_name}.{current if current != 'sql' else 'db'}")
df_current = load_ops[current](file_path)

# Display full loaded data
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
print(df_current)

# --- Analysis & Visualization ---
sns.set(style="whitegrid")

def plot_humidity_temp_precip_linear(df):
    needed = ["high_humidity", "tavg", "prcp"]
    if not all(col in df.columns for col in needed):
        return
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["high_humidity"], label="Humidity (%)", color="green", linestyle='-')
    plt.plot(df.index, df["tavg"], label="Temperature (°C)", color="orange", linestyle='--')
    plt.plot(df.index, df["prcp"], label="Precipitation (mm)", color="blue", linestyle='-.')
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Humidity, Temperature & Precipitation Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_temp_precip_linear(df):
    if "tavg" not in df.columns or "prcp" not in df.columns:
        return
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["tavg"], label="Temperature (°C)", color="red", linestyle="-")
    plt.plot(df.index, df["prcp"], label="Precipitation (mm)", color="blue", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Temperature vs Precipitation Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_thunder_precip_flood_linear(df):
    needed = ["thunderstorm_days", "prcp", "flood_risk"]
    if not all(col in df.columns for col in needed):
        return
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["thunderstorm_days"], label="Thunderstorm Days", color="purple", linestyle='-')
    plt.plot(df.index, df["prcp"], label="Precipitation (mm)", color="blue", linestyle="--")
    plt.plot(df.index, df["flood_risk"], label="Flood Risk Index", color="red", linestyle=":")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Thunderstorm Days, Precipitation & Flood Risk Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# New plot: Thunderstorm Days vs Cyclone Index vs Temperature (3 straight lines)
# Ensure a cyclone_index exists (simulate if not present)
if "cyclone_index" not in df_current.columns:
    df_current["cyclone_index"] = np.random.rand(len(df_current)) * 100  # 0-100 index

def plot_thunder_cyclone_temp(df):
    needed = ["thunderstorm_days", "cyclone_index", "tavg"]
    if not all(col in df.columns for col in needed):
        return
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["thunderstorm_days"], label="Thunderstorm Days", linestyle='-', marker='o')
    plt.plot(df.index, df["cyclone_index"], label="Cyclone Index", linestyle='-', marker='s')
    plt.plot(df.index, df["tavg"], label="Temperature (°C)", linestyle='-', marker='x')
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Thunderstorm Days vs Cyclone Index vs Temperature Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot the graphs
plot_humidity_temp_precip_linear(df_current)
plot_temp_precip_linear(df_current)
plot_thunder_precip_flood_linear(df_current)
plot_thunder_cyclone_temp(df_current)

# --- Flood Risk Prediction (ARIMA) ---
if "flood_risk" in df_current.columns:
    model = ARIMA(df_current["flood_risk"], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(10)
    plt.figure(figsize=(10, 5))
    plt.plot(df_current.index, df_current["flood_risk"], label="Flood Risk (Historical)", color="blue", marker="o")
    # forecast is an array without index; plot it against a simple range on the x-axis to avoid errors
    plt.plot(range(len(df_current.index), len(df_current.index) + len(forecast)), forecast, label="Forecasted Flood Risk", color="red", linestyle="--", marker="x")
    plt.xlabel("Date (index positions for forecast)")
    plt.ylabel("Flood Risk")
    plt.title("Flood Risk Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
