import math
import json
import re
from datetime import datetime, date, timedelta, timezone

import requests
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import matplotlib.pyplot as plt
from matplotlib.table import Table

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.units import inch

from geopy.geocoders import Nominatim

# =========================
# CONFIG
# =========================

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

DAYS_HISTORY_YEARS = 3       # last 3 years
FORECAST_DAYS = 10           # predict next 10 days
RAIN_THRESHOLD_MM = 1.0      # considered a "rainy" day in history

OUT_JSON = "ml_forecast_10d.json"
OUT_PNG = "ml_forecast_table_10d.png"
OUT_PDF = "ml_forecast_report_10d.pdf"

# Geopy Nominatim for precise address geocoding
geolocator = Nominatim(user_agent="weather-ml-geocoder")


# =========================
# UTILS
# =========================

def clean_addr(a):
    a = a.replace("/", " ").replace("-", " ")
    a = re.sub(r"\s+", " ", a)
    return a.strip()


# =========================
# GEOCODING
# =========================

def geocode_open_meteo_raw(name, country_code=None):
    """Call Open-Meteo geocoding once for a given name."""
    params = {
        "name": name,
        "count": 5,
        "language": "en",
        "format": "json",
    }
    if country_code:
        params["country"] = country_code.upper()

    r = requests.get(GEOCODING_URL, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def geocode_address(address, prefer_country_code=None, debug=False):
    """
    1) Try Nominatim on the full address and shorter variants (street -> area -> city -> state -> country)
       to get precise lat/lon.
    2) If Nominatim fails, fall back to Open-Meteo geocoding (city-level).
    """
    cleaned = clean_addr(address)
    if not cleaned:
        raise ValueError("Address is empty after cleaning.")

    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    candidates = []

    # Full address first
    candidates.append(cleaned)

    # Progressive shorter variants from tail:
    # e.g. [25 New Colony, Chrompet, Chennai, Tamil Nadu, India]
    # -> Chrompet, Chennai, Tamil Nadu, India
    # -> Chennai, Tamil Nadu, India
    # -> Tamil Nadu, India
    # -> India
    for n in range(len(parts) - 1, 0, -1):
        tail = ", ".join(parts[-n:])
        if tail not in candidates:
            candidates.append(tail)

    # If "India" not explicitly mentioned, add versions with India
    low = cleaned.lower()
    if "india" not in low and " in" not in low:
        with_india = cleaned + ", India"
        with_in = cleaned + ", IN"
        if with_india not in candidates:
            candidates.append(with_india)
        if with_in not in candidates:
            candidates.append(with_in)

    # Deduplicate
    seen = set()
    uniq_candidates = []
    for c in candidates:
        if c not in seen:
            uniq_candidates.append(c)
            seen.add(c)

    if debug:
        print("[DEBUG] Geocode candidates (Nominatim + Open-Meteo):")
        for c in uniq_candidates:
            print("   -", c)

    # ---------- PASS 1: Nominatim with country filter (if given) ----------
    if prefer_country_code:
        for cand in uniq_candidates:
            try:
                loc = geolocator.geocode(
                    cand,
                    timeout=10,
                    country_codes=prefer_country_code.lower()
                )
            except Exception as e:
                print(f"[WARN] Nominatim (country={prefer_country_code}) failed for '{cand}': {e}")
                continue

            if not loc:
                continue

            # Double-check country
            if loc.raw:
                addr_country = str(
                    loc.raw.get("address", {}).get("country_code", "")
                ).upper()
                if addr_country != prefer_country_code.upper():
                    continue

            return loc.latitude, loc.longitude, loc.address, f"Nominatim (country={prefer_country_code})"

    # ---------- PASS 2: Nominatim with NO country filter ----------
    for cand in uniq_candidates:
        try:
            loc = geolocator.geocode(cand, timeout=10)
        except Exception as e:
            print(f"[WARN] Nominatim (no country) failed for '{cand}': {e}")
            continue

        if loc:
            return loc.latitude, loc.longitude, loc.address, "Nominatim (no country)"

    # ---------- PASS 3: Fallback to Open-Meteo geocoding ----------
    for cand in uniq_candidates:
        try:
            js = geocode_open_meteo_raw(
                cand,
                country_code=prefer_country_code
            )
        except Exception as e:
            print(f"[WARN] Open-Meteo geocoding failed for '{cand}': {e}")
            continue

        results = js.get("results") or []
        if not results:
            continue

        # If prefer_country_code is given, try to pick a matching result
        if prefer_country_code:
            for item in results:
                if str(item.get("country_code", "")).upper() == prefer_country_code.upper():
                    lat = item["latitude"]
                    lon = item["longitude"]
                    name = item.get("name", "")
                    admin1 = item.get("admin1", "")
                    country = item.get("country", "")
                    display = ", ".join([p for p in [name, admin1, country] if p])
                    return lat, lon, display, f"Open-Meteo geocoding (country={prefer_country_code})"

        # Else just take the first result
        first = results[0]
        lat = first["latitude"]
        lon = first["longitude"]
        name = first.get("name", "")
        admin1 = first.get("admin1", "")
        country = first.get("country", "")
        display = ", ".join([p for p in [name, admin1, country] if p])
        return lat, lon, display, "Open-Meteo geocoding"

    # If everything fails:
    raise ValueError(
        "Could not geocode the address using Nominatim or Open-Meteo. "
        "Try a simpler format like 'Chrompet, Chennai, Tamil Nadu, India'."
    )


# =========================
# HISTORICAL DATA FETCH
# =========================

def fetch_history_3years(lat, lon):
    """
    Fetch last 3 years of daily data from Open-Meteo Historical Weather API.
    We use daily mean/max/min temp, precipitation, mean RH, max wind.
    """
    today = date.today()
    start_date = today - timedelta(days=365 * DAYS_HISTORY_YEARS)
    # Historical API has ~2-5 days delay; to be safe, stop 3 days ago
    end_date = today - timedelta(days=3)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join([
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "relative_humidity_2m_mean",
        ]),
        "timezone": "auto",
    }

    r = requests.get(HISTORICAL_URL, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()

    daily = js.get("daily") or {}
    times = daily.get("time") or []
    if not times:
        raise ValueError("Historical API returned no daily data.")

    df = pd.DataFrame({
        "date": pd.to_datetime(times),
        "t_mean": daily.get("temperature_2m_mean"),
        "t_max": daily.get("temperature_2m_max"),
        "t_min": daily.get("temperature_2m_min"),
        "precip": daily.get("precipitation_sum"),
        "wind_max": daily.get("wind_speed_10m_max"),
        "rh_mean": daily.get("relative_humidity_2m_mean"),
    })

    # Basic cleaning
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["t_mean"])  # must have mean temp at least
    return df


# =========================
# ML TRAINING
# =========================

def build_training_data(df):
    """
    From 3y history df, build features/targets for:
      - t_mean_next regression
      - rain_next classification

    Features:
      - doy (day-of-year)
      - sin_doy, cos_doy
      - t_mean (today)

    Targets:
      - t_mean_next: tomorrow's mean temperature
      - rain_next: whether tomorrow is a rainy day (precip >= RAIN_THRESHOLD_MM)
    """
    df = df.copy()
    df["doy"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.0)

    # Rain flag for current day
    df["rain_flag"] = (df["precip"].fillna(0) >= RAIN_THRESHOLD_MM).astype(int)

    # Targets for next day
    df["t_mean_next"] = df["t_mean"].shift(-1)
    df["rain_next"] = df["rain_flag"].shift(-1)

    # Drop last row (no next-day data)
    df_train = df.dropna(subset=["t_mean_next", "rain_next"]).copy()

    X_reg = df_train[["doy", "sin_doy", "cos_doy", "t_mean"]].values
    y_reg = df_train["t_mean_next"].values

    X_cls = df_train[["doy", "sin_doy", "cos_doy"]].values
    y_cls = df_train["rain_next"].astype(int).values

    # Also compute typical deltas to reconstruct t_min/t_max
    delta_max = (df_train["t_max"] - df_train["t_mean"]).dropna().mean()
    delta_min = (df_train["t_mean"] - df_train["t_min"]).dropna().mean()

    if math.isnan(delta_max):
        delta_max = 2.0
    if math.isnan(delta_min):
        delta_min = 2.0

    return X_reg, y_reg, X_cls, y_cls, delta_max, delta_min, df


def train_models(X_reg, y_reg, X_cls, y_cls):
    """
    Train:
      - RandomForestRegressor for t_mean_next
      - RandomForestClassifier for rain_next
    """
    reg = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    reg.fit(X_reg, y_reg)

    cls = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    cls.fit(X_cls, y_cls)

    return reg, cls


# =========================
# ML FORECAST
# =========================

def ml_forecast_next_days(df, reg, cls, delta_max, delta_min, n_days=FORECAST_DAYS):
    """
    Use trained models to forecast next n_days.
    Strategy:
      - Start from last known day
      - For each step:
        * compute doy, sin_doy, cos_doy for forecast_date
        * use reg with [doy, sin_doy, cos_doy, current_t_mean] to predict next_t_mean
        * use cls with [doy, sin_doy, cos_doy] to predict rain probability
        * approximate t_min/t_max from t_mean using historical deltas
    """
    df = df.sort_values("date")
    last_row = df.iloc[-1]
    current_date = last_row["date"].date()
    current_t_mean = float(last_row["t_mean"])

    forecasts = []

    for _ in range(n_days):
        forecast_date = current_date + timedelta(days=1)

        doy = forecast_date.timetuple().tm_yday
        sin_doy = math.sin(2 * math.pi * doy / 365.0)
        cos_doy = math.cos(2 * math.pi * doy / 365.0)

        X_reg_new = np.array([[doy, sin_doy, cos_doy, current_t_mean]])
        t_mean_next = float(reg.predict(X_reg_new)[0])

        X_cls_new = np.array([[doy, sin_doy, cos_doy]])
        # Use predict_proba if available
        if hasattr(cls, "predict_proba"):
            proba = cls.predict_proba(X_cls_new)[0]
            # Assuming class order [0,1], prob_rain = prob(class 1)
            prob_rain = float(proba[1])
        else:
            pred_class = cls.predict(X_cls_new)[0]
            prob_rain = 1.0 if pred_class == 1 else 0.0

        rain_yes = prob_rain >= 0.5

        # Approximate min/max
        t_max_pred = t_mean_next + delta_max
        t_min_pred = t_mean_next - delta_min

        forecasts.append({
            "date": forecast_date.isoformat(),
            "t_mean": t_mean_next,
            "t_max": t_max_pred,
            "t_min": t_min_pred,
            "rain_prob": prob_rain,
            "rain_yes": rain_yes,
        })

        # Move forward
        current_date = forecast_date
        current_t_mean = t_mean_next

    return forecasts


# =========================
# TABLE IMAGE (PNG)
# =========================

def save_color_table(forecast_list, out_png=OUT_PNG):
    headers = ["Date", "Tmean(°C)", "Tmax", "Tmin",
               "RainProb(%)", "Rain?"]

    rows = []
    colors = []

    for r in forecast_list:
        t_mean = r.get("t_mean")
        t_max = r.get("t_max")
        t_min = r.get("t_min")
        rain_prob = r.get("rain_prob", 0.0)
        rain_yes = r.get("rain_yes", False)

        prob_pct = int(round(rain_prob * 100))

        def fmt(x, nd=1):
            if x is None:
                return "-"
            return f"{x:.{nd}f}"

        row = [
            r.get("date", ""),
            fmt(t_mean),
            fmt(t_max),
            fmt(t_min),
            str(prob_pct),
            "Yes" if rain_yes else "No",
        ]
        rows.append(row)

        if t_mean is None or math.isnan(t_mean):
            colors.append(["#ffffff"] * len(headers))
        else:
            if t_mean < 15:
                col = "#cfeefc"
            elif t_mean < 25:
                col = "#fff4cc"
            else:
                col = "#ffd6d6"
            colors.append([col] * len(headers))

    n_rows = len(rows) + 1
    n_cols = len(headers)

    fig, ax = plt.subplots(figsize=(9, 0.6 * n_rows + 1))
    ax.set_axis_off()

    table = Table(ax, bbox=[0, 0, 1, 1])

    col_width = 1.0 / n_cols
    row_height = 1.0 / n_rows

    # Header
    for j, header in enumerate(headers):
        table.add_cell(
            0, j, width=col_width, height=row_height,
            text=header, loc="center", facecolor="#40466e"
        )

    # Rows
    for i, row in enumerate(rows):
        for j, cell_text in enumerate(row):
            cell_color = colors[i][j]
            table.add_cell(
                i + 1, j, width=col_width, height=row_height,
                text=cell_text, loc="center", facecolor=cell_color
            )

    ax.add_table(table)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


# =========================
# PDF REPORT
# =========================

def create_pdf_report(location_display, lat, lon, forecast_list, png_table, out_pdf=OUT_PDF):
    doc = SimpleDocTemplate(out_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title_style = styles["Title"]
    normal_style = styles["Normal"]

    story.append(Paragraph("10-day Weather Forecast (3-year ML Model)", title_style))
    story.append(Spacer(1, 0.2 * inch))

    info_text = f"Location: {location_display} (lat={lat:.4f}, lon={lon:.4f})"
    story.append(Paragraph(info_text, normal_style))
    story.append(Spacer(1, 0.1 * inch))

    gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph(f"Generated at: {gen_time}", normal_style))
    story.append(Spacer(1, 0.2 * inch))

    # Table image
    img_height = 120 + 20 * len(forecast_list)
    img = RLImage(png_table, width=7.2 * inch, height=(img_height / 72.0) * inch)
    story.append(img)
    story.append(Spacer(1, 0.2 * inch))

    # Text summary
    story.append(Paragraph("Daily Summary:", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    for r in forecast_list:
        date_str = r.get("date", "")
        t_mean = r.get("t_mean")
        t_max = r.get("t_max")
        t_min = r.get("t_min")
        rain_prob = r.get("rain_prob", 0.0)
        rain_yes = r.get("rain_yes", False)

        parts = []
        if t_mean is not None:
            parts.append(f"Tmean {t_mean:.1f}°C")
        if t_max is not None:
            parts.append(f"Tmax {t_max:.1f}°C")
        if t_min is not None:
            parts.append(f"Tmin {t_min:.1f}°C")
        parts.append(f"RainProb {int(round(rain_prob * 100))}%")
        parts.append(f"Rain? {'Yes' if rain_yes else 'No'}")

        line = f"{date_str}: " + " | ".join(parts)
        story.append(Paragraph(line, normal_style))
        story.append(Spacer(1, 0.05 * inch))

    doc.build(story)


# =========================
# MAIN
# =========================

def main():
    print("=== 3-year ML Weather Forecast (Open-Meteo + Nominatim) ===")
    addr = input("Enter address (eg. '25 New Colony, Chrompet, Chennai, Tamil Nadu, India'): ").strip()
    if not addr:
        print("No address given, exiting.")
        return

    prefer_IN = (
        "india" in addr.lower()
        or ", in" in addr.lower()
        or addr.strip().lower().endswith(", in")
    )

    try:
        lat, lon, display, source = geocode_address(
            addr,
            prefer_country_code="IN" if prefer_IN else None,
            debug=True
        )
    except Exception as e:
        print(f"Geocoding failed: {e}")
        return

    print(f"\nResolved location: {display}")
    print(f"Coordinates: lat={lat}, lon={lon}")
    print(f"Geocode source: {source}\n")

    print("Fetching last 3 years daily weather data (this may take a few seconds)...")
    try:
        df_hist = fetch_history_3years(lat, lon)
    except Exception as e:
        print(f"Failed to fetch historical data: {e}")
        return

    if len(df_hist) < 365:
        print(f"[WARN] Only {len(df_hist)} days of history available. "
              f"Model may be weak, but continuing.")

    print(f"Loaded {len(df_hist)} days of historical daily data from "
          f"{df_hist['date'].min().date()} to {df_hist['date'].max().date()}.")

    # Build training data
    try:
        X_reg, y_reg, X_cls, y_cls, delta_max, delta_min, df_processed = build_training_data(df_hist)
    except Exception as e:
        print(f"Failed to build training data: {e}")
        return

    # Train models
    print("Training ML models (RandomForest)...")
    try:
        reg, cls = train_models(X_reg, y_reg, X_cls, y_cls)
    except Exception as e:
        print(f"Training failed: {e}")
        return

    # Forecast next days
    print(f"Forecasting next {FORECAST_DAYS} days using the trained models...")
    try:
        forecast_list = ml_forecast_next_days(df_processed, reg, cls, delta_max, delta_min,
                                              n_days=FORECAST_DAYS)
    except Exception as e:
        print(f"Forecasting failed: {e}")
        return

    # Save JSON
    out_obj = {
        "location": {
            "display": display,
            "lat": lat,
            "lon": lon
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "history_start": df_hist["date"].min().date().isoformat(),
        "history_end": df_hist["date"].max().date().isoformat(),
        "forecast": forecast_list
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    # PNG & PDF
    save_color_table(forecast_list, out_png=OUT_PNG)
    create_pdf_report(display, lat, lon, forecast_list, png_table=OUT_PNG, out_pdf=OUT_PDF)

    # Console summary
    print("\n=== ML Forecast Summary ===")
    for r in forecast_list:
        date_str = r["date"]
        t_mean = r.get("t_mean")
        rain_prob = r.get("rain_prob", 0.0)
        rain_yes = r.get("rain_yes", False)

        t_str = "N/A" if t_mean is None else f"{t_mean:.1f}°C"
        prob_pct = int(round(rain_prob * 100))
        emoji = "🌧" if rain_yes else "☀️"

        print(
            f"{date_str}: Tmean={t_str}, RainProb={prob_pct}%, "
            f"Rain?={'Yes' if rain_yes else 'No'} {emoji}"
        )

    print("\nOutputs:")
    print("  JSON :", OUT_JSON)
    print("  PNG  :", OUT_PNG)
    print("  PDF  :", OUT_PDF)
    print("Done.")


if __name__ == "__main__":
    main()
