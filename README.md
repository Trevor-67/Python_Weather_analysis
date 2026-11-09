# Weather Trend Analysis & Flood Risk Prediction

**Python | Data Analysis | Data Visualisation | Time-Series Forecasting**

## Project Overview

Weather Trend Analysis is a Python-based desktop application developed to analyse historical weather patterns, visualise meteorological data, identify temperature trends, and estimate potential flood risk based on precipitation records.

The application uses historical weather observations retrieved through the Meteostat library and applies data analysis and statistical forecasting techniques to help users explore weather patterns.

The project focuses primarily on South African locations, particularly the Eastern Cape, while also supporting custom geographic coordinates.

## Project Objectives

The application aims to:

- Retrieve and analyse historical weather data.
- Identify changes in temperature over time.
- Visualise weather indicators using interactive desktop controls.
- Analyse historical precipitation patterns.
- Identify periods of unusually high rainfall.
- Forecast future precipitation using statistical modelling.
- Support data-driven exploration of weather and environmental trends.

## Key Features

### 1. Weather Data Retrieval

- Retrieves historical daily weather observations using Meteostat.
- Searches nearby weather stations for available data.
- Supports predefined South African cities.
- Allows manual latitude and longitude input.
- Supports user-selected date ranges.

### 2. Data Cleaning and Processing

Uses Pandas to process weather datasets.

Features include:
- Time-based interpolation of missing values.
- Selection of weather indicators.
- Tabular data previews.
- Searching and filtering datasets.
- Paginated data viewing.

### 3. Weather Trend Visualisation

Uses Matplotlib to generate visual representations of historical weather data.

Supports analysis of indicators including:
- Average, minimum and maximum temperature.
- Precipitation.
- Wind speed and direction.
- Air pressure.
- Sunshine duration.
- Snowfall and peak wind gusts, where available.

### 4. Temperature Trend Analysis

Uses NumPy linear regression through `polyfit()` to estimate temperature trends over time.

The application calculates:
- Estimated temperature trend per observation step.
- Approximate annualised temperature trend.
- A fitted trend line for visual comparison with historical data.

These estimates are descriptive and should not be interpreted as definitive evidence of long-term climate change without additional analysis.

### 5. Historical Flood Risk Analysis

Uses precipitation records to identify periods of unusually high accumulated rainfall.

The process:
1. Calculates rolling precipitation totals.
2. Establishes a historical percentile threshold.
3. Flags periods where rainfall exceeds the selected threshold.
4. Visualises the results.

This is a statistical flood-risk indicator, not a validated hydrological flood prediction model.

### 6. Precipitation Forecasting

Uses an ARIMA(2,0,0) statistical model to forecast precipitation.

Features include:
- Configurable forecast duration.
- Forecast visualisation.
- Comparison with historical precipitation thresholds.
- Identification of forecast periods exceeding the selected threshold.

If ARIMA fitting fails, the application uses a simple last-observation forecast as a fallback.

### 7. Data Import and Export

Supports weather dataset import and export using:

| Format | Extension |
|---|---|
| CSV | `.csv` |
| Excel | `.xlsx` |
| JSON | `.json` |
| XML | `.xml` |
| SQLite | `.db` |

The application also attempts to save retrieved weather datasets automatically in multiple formats.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Pandas | Data cleaning and manipulation |
| NumPy | Numerical calculations and trend estimation |
| Matplotlib | Data visualisation |
| Seaborn | Imported visualisation library |
| Statsmodels | ARIMA forecasting |
| Meteostat | Historical weather data retrieval |
| Tkinter | Desktop graphical user interface |
| SQLite | Weather data storage |

## Application Architecture

The project is organised around two main components.

**WeatherDataManager**

Handles:
- Weather station discovery.
- Weather data retrieval.
- Data cleaning.
- Data import and export.
- Temperature trend calculations.
- Historical precipitation risk analysis.
- Precipitation forecasting.

**WeatherAppTk**

Handles:
- Graphical user interface.
- Location and date selection.
- User interactions.
- Data previews.
- Visualisations.
- Forecast and flood-risk controls.

## How to Run the Project

### Prerequisites

- Python 3.10 or later (recommended; compatibility depends on installed libraries).
- Internet access for retrieving weather observations.
- Tkinter support in your Python installation.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

Replace the example URL with your actual repository link.

### Step 2: Navigate to the Project

```bash
cd YOUR-REPOSITORY
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
python -m pip install numpy pandas matplotlib seaborn statsmodels meteostat openpyxl lxml
```

### Step 5: Run the Application

Assuming the main Python file is named `weather_trend_analysis.py`:

```bash
python weather_trend_analysis.py
```

Replace the filename if your project uses a different name.

## How to Use

1. Launch the desktop application.
2. Select a predefined South African city or enter coordinates.
3. Choose a start and end date.
4. Select the weather indicators to retrieve.
5. Click **Fetch (station-aware)** to load historical weather data.
6. View and search the dataset.
7. Select **Visualize** to plot weather indicators.
8. Use **Temp Trend** to analyse temperature changes.
9. Use **Flood Risk (historical)** to identify high-rainfall periods.
10. Use **Forecast & Flood Ahead** to explore future precipitation estimates.
11. Export the data in your preferred format.

## Current Limitations

- Weather data availability depends on nearby stations and Meteostat coverage.
- Some weather indicators may be unavailable for selected locations or dates.
- The flood-risk model uses rainfall thresholds rather than hydrological modelling.
- Forecast accuracy has not been independently validated.
- The application currently uses a desktop interface rather than a web interface.
- The **Analyze** button references an `analyze_results()` method that is not defined in the supplied code and requires correction.
- Additional testing is needed for data import/export compatibility and forecasting edge cases.

## Future Improvements

- Improve forecasting accuracy and model evaluation.
- Add model performance metrics such as MAE and RMSE.
- Expand temperature and seasonal analysis.
- Improve historical flood-risk assessment.
- Integrate geographic visualisations.
- Introduce automated analytical reports.
- Add additional weather data sources.
- Improve error handling and application testing.
- Explore deployment as a web-based analytics dashboard.

## Skills Demonstrated

This project demonstrates practical experience in:

- Python application development.
- Data analysis using Pandas.
- Data cleaning and transformation.
- Numerical analysis using NumPy.
- Data visualisation with Matplotlib.
- Time-series forecasting with ARIMA.
- Working with external weather data sources.
- SQLite data storage.
- Desktop GUI development using Tkinter.
- Applying statistical techniques to real-world environmental data.

## Project Status

**Functional prototype with further testing and improvements planned.**

The project provides a foundation for weather analytics and precipitation-based risk exploration.

**Disclaimer:** Forecasts and flood-risk indicators generated by this application are experimental and should not be used as official weather warnings or emergency decision-making tools.

---

**Project:** Weather Trend Analysis & Flood Risk Prediction  
**Programming Language:** Python  
**Category:** Data Analytics / Statistical Forecasting  
**Application Type:** Desktop Application  
**Focus:** South African Weather Data

