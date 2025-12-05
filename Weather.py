
import os
import sqlite3
import threading
from datetime import datetime
import math
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA

from meteostat import Point, Daily, Stations

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------
# Constants & config
# ---------------------------
INDICATORS_INFO = {
    "tavg": ("Average Temperature", "°C"),
    "tmin": ("Minimum Temperature", "°C"),
    "tmax": ("Maximum Temperature", "°C"),
    "prcp": ("Precipitation", "mm"),
    "snow": ("Snowfall", "mm"),
    "wdir": ("Wind Direction", "°"),
    "wspd": ("Wind Speed", "km/h"),
    "wpgt": ("Peak Wind Gust", "km/h"),
    "pres": ("Air Pressure", "hPa"),
    "tsun": ("Sunshine Duration", "minutes")
}
DEFAULT_SAVE_DIR = "weather_reports"
os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)

# Predefined city -> coords mapping (Eastern Cape + common SA cities)
CITY_COORDS = {
    # Eastern Cape
    "gqeberha": (-33.9613, 25.6122),  # formerly Port Elizabeth
    "port elizabeth": (-33.9613, 25.6122),
    "east london": (-33.0153, 27.9116),
    "mthatha": (-31.5893, 28.7840),
    "king williams town": (-32.8764, 27.4438),
    "bhisho": (-32.8490, 27.3670),
    # Other major SA cities
    "johannesburg": (-26.2041, 28.0473),
    "pretoria": (-25.7479, 28.2293),
    "cape town": (-33.9249, 18.4241),
    "durban": (-29.8587, 31.0218),
    "bloemfontein": (-29.0852, 26.1596),
    "polokwane": (-23.8965, 29.4486),
    "kimberley": (-28.7380, 24.7700),
}

# ---------------------------
# Backend: WeatherDataManager
# ---------------------------
class WeatherDataManager:
    def __init__(self, save_path=DEFAULT_SAVE_DIR):
        self.lat = None
        self.lon = None
        self.start = None
        self.end = None
        self.indicators = []
        self.df_raw = None   # raw DataFrame returned from Meteostat (all cols)
        self.df = None       # filtered (only selected indicators)
        self.df_clean = None
        self.save_path = save_path
        self.last_station = None

    def find_station(self, max_stations=10):
        """Find a nearby station that has daily data for the requested date range."""
        if self.lat is None or self.lon is None or self.start is None or self.end is None:
            return None, None
        try:
            stations = Stations().nearby(self.lat, self.lon)
            near = stations.fetch(max_stations)
        except Exception:
            return None, None
        if near is None or near.empty:
            return None, None
        for _, row in near.iterrows():
            station_id = row.get("id")
            station_name = row.get("name", None)
            try:
                test_df = Daily(station_id, self.start, self.end).fetch()
            except Exception:
                continue
            if test_df is not None and not test_df.empty:
                station_info = {
                    "id": station_id,
                    "name": station_name,
                    "distance": float(row.get("distance")) if "distance" in row.index and pd.notna(row.get("distance")) else None,
                    "country": row.get("country", None),
                    "region": row.get("region", None)
                }
                self.last_station = station_info
                return station_id, station_info
        return None, None

    def fetch_from_cloud(self, lat, lon, start, end, indicators):
        """Fetch data: find nearby station with daily records, fetch, filter indicators."""
        self.lat, self.lon, self.start, self.end = lat, lon, start, end
        self.indicators = indicators

        station_id, station_info = self.find_station(max_stations=12)
        if station_id is None:
            # fallback to Point-based Daily
            try:
                location = Point(lat, lon)
                df_full = Daily(location, start, end).fetch()
            except Exception:
                df_full = pd.DataFrame()
            available_cols = list(df_full.columns) if hasattr(df_full, "columns") else []
            missing = [c for c in indicators if c not in available_cols]
            selected = [c for c in indicators if c in available_cols]
            if not selected:
                df_selected = pd.DataFrame(index=df_full.index if hasattr(df_full, "index") else None)
            else:
                df_selected = df_full[selected].copy()
            self.df_raw = df_full.copy() if hasattr(df_full, "copy") else pd.DataFrame()
            self.df = df_selected
            info = {"available_cols": available_cols, "missing": missing, "selected": selected, "station": station_info}
            return df_selected, info

        # fetch by station id
        try:
            df_full = Daily(station_id, start, end).fetch()
            if df_full is None:
                df_full = pd.DataFrame()
        except Exception:
            df_full = pd.DataFrame()

        self.df_raw = df_full.copy()
        available_cols = list(df_full.columns)
        missing = [c for c in indicators if c not in available_cols]
        selected = [c for c in indicators if c in available_cols]
        if not selected:
            df_selected = pd.DataFrame(index=df_full.index)
        else:
            df_selected = df_full[selected].copy()

        self.df = df_selected
        info = {"available_cols": available_cols, "missing": missing, "selected": selected, "station": station_info}
        return df_selected, info

    def clean(self):
        if self.df is None:
            raise ValueError("No data to clean.")
        if self.df.empty:
            raise ValueError("Data is empty; nothing to clean.")
        self.df_clean = self.df.interpolate(method='time')
        return self.df_clean

    def save(self, path, fmt):
        if self.df is None or (hasattr(self.df, "empty") and self.df.empty):
            raise ValueError("No data to save.")
        fmt = fmt.lower()
        if fmt == "excel": fmt = "xlsx"
        if fmt == 'csv':
            self.df.to_csv(path)
        elif fmt == 'xlsx':
            self.df.to_excel(path)
        elif fmt == 'json':
            self.df.to_json(path, orient='records', date_format='iso')
        elif fmt == 'xml':
            self.df.to_xml(path, root_name='weather', row_name='day')
        elif fmt == 'sql':
            conn = sqlite3.connect(path)
            try:
                self.df.to_sql('weather', conn, if_exists='replace', index=True)
            finally:
                conn.close()
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def load_local(self, path, fmt):
        fmt = fmt.lower()
        if fmt == "excel":
            fmt = "xlsx"
        if fmt == 'csv':
            df = pd.read_csv(path, parse_dates=True, index_col=0)
        elif fmt == 'xlsx':
            df = pd.read_excel(path, parse_dates=True, index_col=0)
        elif fmt == 'json':
            df = pd.read_json(path).set_index('time'); df.index = pd.to_datetime(df.index)
        elif fmt == 'xml':
            df = pd.read_xml(path).set_index('time'); df.index = pd.to_datetime(df.index)
        elif fmt == 'sql':
            conn = sqlite3.connect(path)
            df = pd.read_sql('SELECT * FROM weather', conn, parse_dates=['time'], index_col='time')
            conn.close()
        else:
            raise ValueError("Unsupported format for loading.")
        self.df = df
        return df

    # -------------------------
    # Flood-risk and trends
    # -------------------------
    def flood_risk(self, prcp_col='prcp', window_days=3, percentile=0.95):
        """
        Heuristic flood detection:
        - Compute rolling sum of precipitation over window_days.
        - Compute historical percentile threshold from entire dataset.
        - Flag dates where rolling-sum >= threshold.
        Returns DataFrame with rolling_sum and boolean 'flood_risk'.
        """
        if self.df is None or prcp_col not in self.df.columns:
            raise ValueError("Precipitation data not available for flood analysis.")
        series = self.df[prcp_col].dropna().astype(float)
        if series.empty:
            raise ValueError("No precipitation data.")
        rolling = series.rolling(window=window_days, min_periods=1).sum()
        threshold = rolling.quantile(percentile)
        risk = rolling >= threshold
        out = pd.DataFrame({'prcp_rolling_sum': rolling, 'flood_risk': risk})
        return out, float(threshold)

    def forecast_precip(self, prcp_col='prcp', steps=7):
        """Forecast precipitation (simple ARIMA) and return forecast Series."""
        if self.df is None or prcp_col not in self.df.columns:
            raise ValueError("Precipitation data not available for forecast.")
        series = self.df[prcp_col].dropna().astype(float)
        if series.empty:
            raise ValueError("No precipitation data to forecast.")
        try:
            model = ARIMA(series, order=(2, 0, 0))
            fit = model.fit()
            forecast = fit.forecast(steps=steps)
        except Exception:
            # fallback: naive last value repeat
            last = series.iloc[-1]
            forecast = pd.Series([last]*steps, index=pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=steps))
        # ensure index
        if not isinstance(forecast.index, pd.DatetimeIndex):
            forecast.index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=len(forecast))
        return forecast

    def temperature_trend(self, tcol='tavg'):
        """
        Compute simple linear trend on tcol using numpy polyfit: returns slope (deg per day)
        and intercept plus fit series for plotting.
        """
        if self.df is None or tcol not in self.df.columns:
            raise ValueError("Temperature data not available.")
        series = self.df[tcol].dropna().astype(float)
        if series.empty:
            raise ValueError("No temperature data.")
        x = np.arange(len(series))
        y = series.values
        # linear fit
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        fitted_series = pd.Series(index=series.index, data=fitted)
        # convert slope to deg per year approx:
        slope_per_day = float(slope)
        slope_per_year = slope_per_day * 365.25
        return {'slope_per_day': slope_per_day, 'slope_per_year': slope_per_year, 'fitted': fitted_series}

    # small helper
    def get_numeric_columns(self):
        if self.df_clean is None:
            return []
        return self.df_clean.select_dtypes(include='number').columns.tolist()

# ---------------------------
# UI: Scrollable Frame helper
# ---------------------------
class VerticalScrolledFrame(ttk.Frame):
    def __init__(self, parent, *args, **kw):
        super().__init__(parent, *args, **kw)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)
        self.v_scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.interior = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0,0), window=self.interior, anchor="nw")
        self.interior.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.interior.bind_all("<MouseWheel>", self._on_mousewheel_windows)
        self.interior.bind_all("<Button-4>", self._on_mousewheel_unix)
        self.interior.bind_all("<Button-5>", self._on_mousewheel_unix)

    def _on_canvas_configure(self, event):
        canvas_width = event.width
        self.canvas.itemconfigure(self.window_id, width=canvas_width)

    def _on_mousewheel_windows(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_mousewheel_unix(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

# ---------------------------
# Main App with flood & trend UI additions
# ---------------------------
class WeatherAppTk:
    def __init__(self, root):
        self.root = root
        self.root.title("Weather & Flood Risk App — Eastern Cape Focus")
        self.root.geometry("1200x820")
        self.manager = WeatherDataManager()
        self.fig_canvas = None
        self.current_fig = None
        self._figures = []
        self._fig_idx = -1
        self._setup_ui()
        self.page_size = 50
        self.page_idx = 0
        self._current_preview_df = pd.DataFrame()

    def _setup_ui(self):
        toolbar = ttk.Frame(self.root, padding=6); toolbar.pack(side="top", fill="x")
        ttk.Label(toolbar, text="Weather & Flood Risk App", font=("Segoe UI", 14, "bold")).pack(side="left")

        scrolled = VerticalScrolledFrame(self.root); scrolled.pack(side="top", fill="both", expand=True, padx=6, pady=6)
        frm = scrolled.interior

        # City input
        section_city = ttk.Labelframe(frm, text="Location", padding=8); section_city.pack(fill="x", padx=4, pady=6)
        ttk.Label(section_city, text="City name (Eastern Cape or SA):").grid(row=0, column=0, sticky="w")
        self.city_var = tk.StringVar(value="gqeberha")
        ttk.Entry(section_city, width=28, textvariable=self.city_var).grid(row=0, column=1, padx=6, sticky="w")
        ttk.Button(section_city, text="Use city coords", command=self._on_use_city).grid(row=0, column=2, padx=6)
        ttk.Label(section_city, text="or enter Latitude / Longitude:").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.lat_var = tk.StringVar(value="")
        ttk.Entry(section_city, width=12, textvariable=self.lat_var).grid(row=1, column=1, sticky="w")
        self.lon_var = tk.StringVar(value="")
        ttk.Entry(section_city, width=12, textvariable=self.lon_var).grid(row=1, column=2, sticky="w")

        # Dates & indicators
        section_in = ttk.Labelframe(frm, text="Dates & Indicators", padding=8); section_in.pack(fill="x", padx=4, pady=6)
        ttk.Label(section_in, text="Start date (YYYY-MM-DD)").grid(row=0, column=0, sticky="w")
        self.start_var = tk.StringVar(value=(datetime.now().replace(year=datetime.now().year - 3)).strftime("%Y-%m-%d"))
        ttk.Entry(section_in, width=14, textvariable=self.start_var).grid(row=0, column=1, padx=6, sticky="w")
        ttk.Label(section_in, text="End date (YYYY-MM-DD)").grid(row=0, column=2, sticky="w")
        self.end_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(section_in, width=14, textvariable=self.end_var).grid(row=0, column=3, padx=6, sticky="w")

        ttk.Label(section_in, text="Indicators:").grid(row=1, column=0, sticky="w", pady=(8,0))
        self.indicator_vars = {}
        ind_frame = ttk.Frame(section_in); ind_frame.grid(row=2, column=0, columnspan=4, sticky="w")
        default_checked = set(["tavg", "prcp"])
        r = 0; c = 0
        for k, (lbl, unit) in INDICATORS_INFO.items():
            var = tk.BooleanVar(value=(k in default_checked))
            self.indicator_vars[k] = var
            cb = ttk.Checkbutton(ind_frame, text=f"{k} — {lbl} ({unit})", variable=var)
            cb.grid(row=r, column=c, sticky="w", padx=6, pady=2)
            c += 1
            if c >= 2:
                c = 0; r += 1

        # Operations
        section_ops = ttk.Labelframe(frm, text="Operations", padding=8); section_ops.pack(fill="x", padx=4, pady=6)
        self.fetch_btn = ttk.Button(section_ops, text="Fetch (station-aware)", command=self._on_fetch); self.fetch_btn.grid(row=0, column=0, padx=6)
        self.save_btn = ttk.Button(section_ops, text="Save", command=self._on_save); self.save_btn.grid(row=0, column=1, padx=6)
        self.load_btn = ttk.Button(section_ops, text="Load", command=self._on_load); self.load_btn.grid(row=0, column=2, padx=6)
        ttk.Label(section_ops, text="Save format:").grid(row=0, column=3, padx=(12,4))
        self.save_format_var = tk.StringVar(value="csv")
        ttk.Combobox(section_ops, textvariable=self.save_format_var, values=["csv","xlsx","json","xml","sql"], state="readonly", width=8).grid(row=0, column=4)

        # Station info
        station_frame = ttk.Frame(section_ops); station_frame.grid(row=0, column=5, padx=12)
        ttk.Label(station_frame, text="Station:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.station_info_var = tk.StringVar(value="(none)"); ttk.Label(station_frame, textvariable=self.station_info_var, wraplength=420).pack(anchor="w")
        self.station_cols_var = tk.StringVar(value=""); ttk.Label(station_frame, textvariable=self.station_cols_var, wraplength=420).pack(anchor="w")

        # Preview
        section_prev = ttk.Labelframe(frm, text="Preview", padding=8); section_prev.pack(fill="both", padx=4, pady=6, expand=True)
        search_frame = ttk.Frame(section_prev); search_frame.pack(fill="x")
        ttk.Label(search_frame, text="Search:").pack(side="left")
        self.search_var = tk.StringVar(); ttk.Entry(search_frame, textvariable=self.search_var, width=30).pack(side="left", padx=6)
        ttk.Button(search_frame, text="Apply", command=self._on_search).pack(side="left", padx=6)
        ttk.Button(search_frame, text="Clear", command=self._on_search_clear).pack(side="left", padx=6)
        preview_area = ttk.Frame(section_prev); preview_area.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(preview_area, columns=(), show=""); self.tree.pack(side="left", fill="both", expand=True)
        vsb = ttk.Scrollbar(preview_area, orient="vertical", command=self.tree.yview); vsb.pack(side="right", fill="y"); self.tree.configure(yscrollcommand=vsb.set)
        page_ctrl = ttk.Frame(section_prev); page_ctrl.pack(fill="x")
        self.prev_page_btn = ttk.Button(page_ctrl, text="Previous Page", command=self._prev_page); self.prev_page_btn.pack(side="left", padx=6)
        self.next_page_btn = ttk.Button(page_ctrl, text="Next Page", command=self._next_page); self.next_page_btn.pack(side="left", padx=6)
        self.page_label_var = tk.StringVar(value="Page 0 / 0"); ttk.Label(page_ctrl, textvariable=self.page_label_var).pack(side="left", padx=12)

        # Plots and flood controls
        section_plot = ttk.Labelframe(frm, text="Plots, Trend & Flood Risk", padding=8); section_plot.pack(fill="both", padx=4, pady=6, expand=True)
        plot_ctrl = ttk.Frame(section_plot); plot_ctrl.pack(fill="x")
        ttk.Button(plot_ctrl, text="Visualize", command=self._on_visualize).pack(side="left", padx=6)
        ttk.Button(plot_ctrl, text="Analyze", command=self._on_analyze).pack(side="left", padx=6)
        ttk.Button(plot_ctrl, text="Temp Trend", command=self._on_temp_trend).pack(side="left", padx=6)
        ttk.Button(plot_ctrl, text="Flood Risk (historical)", command=self._on_flood).pack(side="left", padx=6)
        ttk.Button(plot_ctrl, text="Forecast & Flood Ahead", command=self._on_forecast_flood).pack(side="left", padx=6)

        # forecast options
        ttk.Label(plot_ctrl, text="Forecast days:").pack(side="left", padx=(12,2))
        self.forecast_days_var = tk.StringVar(value="7"); ttk.Entry(plot_ctrl, width=6, textvariable=self.forecast_days_var).pack(side="left")
        ttk.Label(plot_ctrl, text="Flood window days:").pack(side="left", padx=(12,2))
        self.flood_win_var = tk.StringVar(value="3"); ttk.Entry(plot_ctrl, width=6, textvariable=self.flood_win_var).pack(side="left")
        ttk.Label(plot_ctrl, text="Percentile:").pack(side="left", padx=(12,2))
        self.flood_pct_var = tk.StringVar(value="0.95"); ttk.Entry(plot_ctrl, width=6, textvariable=self.flood_pct_var).pack(side="left")

        # chart nav
        chart_nav = ttk.Frame(section_plot); chart_nav.pack(fill="x", pady=(6,0))
        self.fig_prev_btn = ttk.Button(chart_nav, text="<< Prev Chart", command=self._prev_fig); self.fig_prev_btn.pack(side="left", padx=6)
        self.fig_next_btn = ttk.Button(chart_nav, text="Next Chart >>", command=self._next_fig); self.fig_next_btn.pack(side="left", padx=6)

        self.plot_container = ttk.Frame(section_plot); self.plot_container.pack(fill="both", expand=True)

        # bottom
        bottom = ttk.Frame(frm); bottom.pack(fill="x", pady=(8,0))
        self.status_var = tk.StringVar(value="Status: idle"); ttk.Label(bottom, textvariable=self.status_var).pack(side="left")
        self.log_text = tk.Text(bottom, height=5, state="disabled", wrap="word"); self.log_text.pack(side="right", fill="x", expand=True)

    # --------------------
    # UI helpers
    # --------------------
    def _log(self, msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_status(self, text):
        self.status_var.set(f"Status: {text}")
        self.root.update_idletasks()

    def _on_use_city(self):
        city = self.city_var.get().strip().lower()
        if city in CITY_COORDS:
            lat, lon = CITY_COORDS[city]
            self.lat_var.set(str(lat))
            self.lon_var.set(str(lon))
            self._log(f"City '{city}' mapped to coords: {lat}, {lon}")
        else:
            messagebox.showinfo("City not found", f"City '{city}' not in built-in list. You can enter coordinates manually.")
            self._log(f"City '{city}' not recognized.")

    def _get_selected_indicators(self):
        return [k for k, v in self.indicator_vars.items() if v.get()]

    def _clear_tree(self):
        for row in self.tree.get_children(): self.tree.delete(row)
        self.tree["columns"] = (); self.tree["show"] = ""

    def _populate_tree_page(self, df):
        self._clear_tree()
        if df is None or df.empty:
            self.page_label_var.set("Page 0 / 0"); return
        start = self.page_idx * self.page_size; end = start + self.page_size
        page_df = df.iloc[start:end].copy().reset_index()
        cols = list(page_df.columns)
        self.tree["columns"] = cols; self.tree["show"] = "headings"
        for c in cols:
            self.tree.heading(c, text=c); self.tree.column(c, width=140, anchor="w")
        for _, row in page_df.iterrows():
            vals = [self._format_val(v) for v in row.tolist()]
            self.tree.insert("", "end", values=vals)
        total_pages = math.ceil(len(df) / self.page_size) if len(df) > 0 else 0
        self.page_label_var.set(f"Page {self.page_idx+1} / {total_pages}")

    @staticmethod
    def _format_val(v):
        if pd.isna(v): return ""
        if isinstance(v, (pd.Timestamp, datetime)): return v.strftime("%Y-%m-%d")
        return str(v)

    def _clear_plot(self):
        if self.fig_canvas:
            try: self.fig_canvas.get_tk_widget().destroy()
            except Exception: pass
            self.fig_canvas = None
        plt.close('all'); self.current_fig = None

    def _embed_figure(self, fig):
        self._clear_plot()
        self.current_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        canvas.draw(); widget = canvas.get_tk_widget(); widget.pack(fill="both", expand=True)
        self.fig_canvas = canvas

    def _show_figure_by_index(self, idx):
        if not self._figures: return
        idx = idx % len(self._figures)
        fig = self._figures[idx]
        self._embed_figure(fig)
        self._fig_idx = idx
        self._update_chart_nav_buttons()

    def _next_fig(self):
        if not self._figures: return
        self._fig_idx = (self._fig_idx + 1) % len(self._figures)
        self._show_figure_by_index(self._fig_idx)

    def _prev_fig(self):
        if not self._figures: return
        self._fig_idx = (self._fig_idx - 1) % len(self._figures)
        self._show_figure_by_index(self._fig_idx)

    def _update_chart_nav_buttons(self):
        n = len(self._figures)
        state_prev = tk.NORMAL if n>0 else tk.DISABLED
        state_next = tk.NORMAL if n>0 else tk.DISABLED
        self.fig_prev_btn.config(state=state_prev); self.fig_next_btn.config(state=state_next)

    # --------------------
    # Fetch / Load / Save
    # --------------------
    def _on_fetch(self):
        try:
            lat = self._to_float(self.lat_var.get(), "Latitude")
            lon = self._to_float(self.lon_var.get(), "Longitude")
            start = self._to_date(self.start_var.get(), "Start date")
            end = self._to_date(self.end_var.get(), "End date")
            if end < start: raise ValueError("End date must be same or after start date.")
            indicators = self._get_selected_indicators()
            if not indicators: raise ValueError("Select at least one indicator.")
        except Exception as e:
            messagebox.showerror("Input error", str(e)); return
        thread = threading.Thread(target=self._fetch_bg, args=(lat, lon, start, end, indicators), daemon=True)
        thread.start()

    def _fetch_bg(self, lat, lon, start, end, indicators):
        self._set_status("fetching..."); self._log(f"Fetching {indicators} at ({lat},{lon}) {start.date()} to {end.date()} ...")
        try:
            df, info = self.manager.fetch_from_cloud(lat, lon, start, end, indicators)
            self._log(f"Available columns: {info['available_cols']}")
            if info['missing']: self._log(f"Missing: {info['missing']}")
            station = info.get("station")
            if station:
                s_txt = f"{station.get('name') or '(no name)'} (id={station.get('id')})"
                if station.get('distance') is not None: s_txt += f" — {station.get('distance'):.1f} km"
                self.station_info_var.set(s_txt)
            else:
                self.station_info_var.set("(no station found)")
            self.station_cols_var.set(", ".join(info.get("available_cols", [])[:30]) or "(none)")
            if df is None or df.empty:
                self._log("Resulting dataframe is empty.")
                messagebox.showwarning("No data", "No data returned. Check station & date range.")
                self._set_status("idle"); self._current_preview_df = pd.DataFrame(); self.page_idx = 0; self._populate_tree_page(self._current_preview_df); return
            # auto-check indicators
            available_selected = info.get("selected", [])
            for k, var in self.indicator_vars.items(): var.set(k in available_selected)
            self.manager.df = df; self._current_preview_df = df.copy(); self.page_idx = 0; self._populate_tree_page(self._current_preview_df)
            try:
                _ = self.manager.clean(); numeric_cols = self.manager.get_numeric_columns()
            except Exception:
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
            self._update_forecast_combobox(numeric_cols)
            self._log(f"Fetched {len(df)} rows, columns: {list(df.columns)}")
            # auto-save all formats
            try:
                base_name = self._make_autosave_basename(lat, lon, start, end)
                self._auto_save_all_formats(base_name)
            except Exception as e:
                self._log("Auto-save error: " + str(e))
            self._set_status("fetched")
        except Exception as e:
            self._log("Fetch error: " + str(e)); messagebox.showerror("Fetch error", str(e)); self._set_status("error")

    def _make_autosave_basename(self, lat, lon, start, end):
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        lat_s = str(lat).replace(".", "_").replace("-", "m"); lon_s = str(lon).replace(".", "_").replace("-", "m")
        start_s = start.strftime("%Y%m%d"); end_s = end.strftime("%Y%m%d")
        base = f"weather_{lat_s}_{lon_s}_{start_s}_{end_s}_{ts}"
        base = "".join(ch for ch in base if (ch.isalnum() or ch in ("_", "-")))
        return base

    def _auto_save_all_formats(self, base_name):
        if self.manager.df is None or self.manager.df.empty:
            self._log("Auto-save skipped: no data to save."); return
        save_dir = self.manager.save_path or DEFAULT_SAVE_DIR; os.makedirs(save_dir, exist_ok=True)
        targets = {
            "csv": os.path.join(save_dir, f"{base_name}.csv"),
            "xlsx": os.path.join(save_dir, f"{base_name}.xlsx"),
            "json": os.path.join(save_dir, f"{base_name}.json"),
            "xml": os.path.join(save_dir, f"{base_name}.xml"),
            "sql": os.path.join(save_dir, f"{base_name}.db")
        }
        for fmt, path in targets.items():
            try:
                self.manager.save(path, fmt); self._log(f"Auto-saved {fmt.upper()} -> {path}")
            except Exception as e:
                self._log(f"Auto-save FAILED for {fmt.upper()} -> {path} : {e}")

    def _on_load(self):
        filetypes = [("CSV","*.csv"),("Excel","*.xlsx;*.xls"),("JSON","*.json"),("XML","*.xml"),("SQLite DB","*.db;*.sqlite"),("All files","*.*")]
        path = filedialog.askopenfilename(title="Open data file", filetypes=filetypes)
        if not path: return
        _, ext = os.path.splitext(path); ext = ext.lower().lstrip("."); fmt = "sql" if ext in ("db","sqlite") else ext
        try:
            df = self.manager.load_local(path, fmt)
            if df is None or df.empty: messagebox.showwarning("Load", "Loaded file is empty.")
            self._current_preview_df = df.copy(); self.page_idx = 0; self._populate_tree_page(self._current_preview_df)
            numeric_cols = df.select_dtypes(include='number').columns.tolist(); self._update_forecast_combobox(numeric_cols)
            self._log(f"Loaded file: {path} ({len(df)} rows, {len(df.columns)} cols)"); self._set_status("loaded")
        except Exception as e:
            self._log("Load error: " + str(e)); messagebox.showerror("Load error", str(e)); self._set_status("error")

    def _on_save(self):
        if self.manager.df is None or self.manager.df.empty:
            messagebox.showwarning("Save", "No data to save."); return
        fmt = self.save_format_var.get().strip().lower();
        if fmt == "excel": fmt = "xlsx"
        default_name = "weather_data"; ext = fmt
        path = filedialog.asksaveasfilename(defaultextension=f".{ext}", initialfile=default_name, filetypes=[(f"{fmt.upper()} files", f"*.{ext}"),("All files","*.*")])
        if not path: return
        try:
            self.manager.save(path, fmt); self._log(f"Saved data to {path}"); self._set_status("saved"); messagebox.showinfo("Save", f"Saved to {path}")
        except Exception as e:
            self._log("Save error: " + str(e)); messagebox.showerror("Save error", str(e)); self._set_status("error")

    # --------------------
    # Search & pagination
    # --------------------
    def _on_search(self):
        query = self.search_var.get().strip().lower()
        if self.manager.df is None or self.manager.df.empty: messagebox.showinfo("Search", "No data to search."); return
        df = self.manager.df.copy()
        if query == "": self._current_preview_df = df
        else:
            mask = pd.Series([False]*len(df), index=df.index)
            for col in df.columns:
                mask = mask | df[col].astype(str).str.lower().str.contains(query, na=False)
            self._current_preview_df = df[mask]
        self.page_idx = 0; self._populate_tree_page(self._current_preview_df); self._log(f"Search: '{query}' -> {len(self._current_preview_df)} rows")

    def _on_search_clear(self):
        self.search_var.set("");
        if self.manager.df is not None:
            self._current_preview_df = self.manager.df.copy(); self.page_idx = 0; self._populate_tree_page(self._current_preview_df); self._log("Search cleared")

    def _prev_page(self):
        if self._current_preview_df is None or self._current_preview_df.empty: return
        if self.page_idx > 0: self.page_idx -= 1; self._populate_tree_page(self._current_preview_df)

    def _next_page(self):
        if self._current_preview_df is None or self._current_preview_df.empty: return
        total_pages = math.ceil(len(self._current_preview_df)/self.page_size)
        if self.page_idx < total_pages-1: self.page_idx += 1; self._populate_tree_page(self._current_preview_df)

    # --------------------
    # Plot / analyze / flood / forecast
    # --------------------
    def _update_forecast_combobox(self, numeric_cols):
        try:
            self.forecast_col = numeric_cols[0] if numeric_cols else ""
        except Exception:
            self.forecast_col = ""

    def _on_visualize(self):
        if self.manager.df is None or self.manager.df.empty:
            messagebox.showwarning("Visualize", "No data available."); return
        try:
            self.manager.clean()
        except Exception as e:
            messagebox.showerror("Clean error", str(e)); return
        df = self.manager.df_clean
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            messagebox.showinfo("Visualize", "No numeric columns."); return
        fig, ax = plt.subplots(figsize=(10,6))
        for col in numeric_cols:
            name, unit = INDICATORS_INFO.get(col, (col, ""))
            ax.plot(df.index, df[col], label=f"{name} ({unit})")
        ax.set_title("Weather indicators over time"); ax.set_xlabel("Date"); ax.legend()
        self._figures = [fig]; self._fig_idx = 0; self._show_figure_by_index(0); self._set_status("visualized"); self._log("Visualized time series")

    def _on_analyze(self):
        if self.manager.df is None or self.manager.df.empty:
            messagebox.showwarning("Analyze", "No data available."); return
        try:
            self.manager.clean()
        except Exception as e:
            messagebox.showerror("Clean error", str(e)); return
        try:
            figs = self.manager.analyze_results()
            if not figs: messagebox.showinfo("Analyze", "No numeric columns."); return
            self._figures = figs; self._fig_idx = 0; self._show_figure_by_index(0); self._set_status("analysis ready"); self._log(f"Analysis generated {len(figs)} figures")
        except Exception as e:
            self._log("Analyze error: " + str(e)); messagebox.showerror("Analyze error", str(e))

    def _on_temp_trend(self):
        if self.manager.df is None or 'tavg' not in (self.manager.df.columns if self.manager.df is not None else []):
            messagebox.showwarning("Temp Trend", "tavg not available."); return
        try:
            self.manager.clean()
            tr = self.manager.temperature_trend('tavg')
            fitted = tr['fitted']; slope_year = tr['slope_per_year']
            fig, ax = plt.subplots(figsize=(10,6))
            series = self.manager.df_clean['tavg'].dropna().astype(float)
            ax.plot(series.index, series, label='tavg (historical)')
            ax.plot(fitted.index, fitted, label=f'Linear fit (slope ≈ {slope_year:.3f} °C/year)', linewidth=2)
            ax.set_title("Temperature trend (global warming indicator)"); ax.set_xlabel("Date"); ax.legend()
            self._figures = [fig]; self._fig_idx = 0; self._show_figure_by_index(0)
            self._set_status("temp trend ready"); self._log(f"Temp trend slope ≈ {slope_year:.4f} °C/year")
        except Exception as e:
            self._log("Temp trend error: " + str(e)); messagebox.showerror("Temp trend error", str(e))

    def _on_flood(self):
        if self.manager.df is None or 'prcp' not in (self.manager.df.columns if self.manager.df is not None else []):
            messagebox.showwarning("Flood Risk", "prcp not available."); return
        try:
            self.manager.clean()
            win = int(self.flood_win_var.get()); pct = float(self.flood_pct_var.get())
            out, threshold = self.manager.flood_risk(prcp_col='prcp', window_days=win, percentile=pct)
            # Plot rolling sum and mark risks
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(out.index, out['prcp_rolling_sum'], label=f"{win}-day rolling precipitation (mm)")
            ax.axhline(threshold, color='red', linestyle='--', label=f"{int(pct*100)}th percentile threshold ({threshold:.2f} mm)")
            ax.fill_between(out.index, out['prcp_rolling_sum'], threshold, where=out['prcp_rolling_sum']>=threshold, color='red', alpha=0.3)
            ax.set_title("Historical flood-risk heuristic (precip rolling-sum)"); ax.set_xlabel("Date"); ax.legend()
            self._figures = [fig]; self._fig_idx = 0; self._show_figure_by_index(0)
            risk_count = int(out['flood_risk'].sum())
            self._log(f"Found {risk_count} high-risk windows historically (threshold={threshold:.2f})"); self._set_status("flood analysis complete")
        except Exception as e:
            self._log("Flood analysis error: " + str(e)); messagebox.showerror("Flood error", str(e))

    def _on_forecast_flood(self):
        if self.manager.df is None or 'prcp' not in (self.manager.df.columns if self.manager.df is not None else []):
            messagebox.showwarning("Forecast Flood", "prcp not available."); return
        try:
            self.manager.clean()
            days = int(self.forecast_days_var.get().strip()); win = int(self.flood_win_var.get()); pct = float(self.flood_pct_var.get())
            forecast = self.manager.forecast_precip(prcp_col='prcp', steps=days)
            # compute rolling sum on historical series to get threshold
            hist_roll = self.manager.df['prcp'].dropna().astype(float).rolling(window=win, min_periods=1).sum()
            threshold = float(hist_roll.quantile(pct))
            # compute forecast rolling sums (for short windows, simple moving sum on forecast)
            # append last (win-1) days from history to forecast to compute rolling sums properly
            tail = self.manager.df['prcp'].dropna().astype(float).iloc[-(win-1):] if len(self.manager.df['prcp'].dropna()) >= (win-1) else self.manager.df['prcp'].dropna()
            combined = pd.concat([tail, forecast])
            f_roll = combined.rolling(window=win, min_periods=1).sum().iloc[len(tail):]
            risk_flags = f_roll >= threshold
            # plot forecast and highlight flood-risk days
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(forecast.index, forecast.values, label='Forecast prcp (mm)')
            ax.plot(f_roll.index, f_roll.values, label=f'{win}-day rolling sum (forecast)')
            ax.axhline(threshold, color='red', linestyle='--', label=f'Historical {int(pct*100)}th pct threshold ({threshold:.2f})')
            ax.scatter(f_roll.index[risk_flags], f_roll[risk_flags], color='red', zorder=5, label='Flood risk predicted')
            ax.set_title("Precipitation forecast & predicted flood-risk"); ax.set_xlabel("Date"); ax.legend()
            self._figures = [fig]; self._fig_idx = 0; self._show_figure_by_index(0)
            predicted = list(f_roll.index[risk_flags].strftime("%Y-%m-%d"))
            self._log(f"Predicted flood-risk dates in forecast: {predicted if predicted else 'None'}")
            self._set_status("forecast ready")
        except Exception as e:
            self._log("Forecast flood error: " + str(e)); messagebox.showerror("Forecast error", str(e))

    def _on_save_figure(self):
        if self.current_fig is None:
            messagebox.showinfo("Save Figure", "No figure displayed."); return
        filetypes = [("PNG","*.png"),("PDF","*.pdf")]
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=filetypes, initialdir=".")
        if not path: return
        try:
            if isinstance(self.current_fig, plt.Figure):
                self.current_fig.savefig(path, bbox_inches="tight")
            else:
                self.current_fig.savefig(path, bbox_inches="tight")
            self._log(f"Figure saved to {path}"); messagebox.showinfo("Saved", f"Figure saved to {path}")
        except Exception as e:
            self._log("Save figure error: " + str(e)); messagebox.showerror("Save figure error", str(e))

    # --------------------
    # Validators
    # --------------------
    def _to_float(self, s, name="value"):
        s = s.strip().replace(",", ".")
        if s == "": raise ValueError(f"{name} is required.")
        try: return float(s)
        except Exception: raise ValueError(f"{name} must be a number (example: -33.96).")

    def _to_date(self, s, name="date"):
        s = s.strip()
        if s == "": raise ValueError(f"{name} is required.")
        try: return datetime.fromisoformat(s)
        except Exception: raise ValueError(f"{name} must be in YYYY-MM-DD format (example: 2022-01-01).")

# ---------------------------
# Run app
# ---------------------------
def main():
    root = tk.Tk()
    app = WeatherAppTk(root)
    root.mainloop()

if __name__ == "__main__":
    main()
