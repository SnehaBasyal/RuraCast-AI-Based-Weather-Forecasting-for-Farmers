# weather_utils.py
import os
import uuid
from datetime import datetime, date, timedelta
import math
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.table import Table
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.units import inch
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import requests
from geopy.geocoders import Nominatim
import time

# Constants
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
DAYS_HISTORY_YEARS = 5
RAIN_THRESHOLD_MM = 1.0
geolocator = Nominatim(user_agent="farmer_advisory_app")

def geocode_address(address):
    """Geocode the address using multiple methods"""
    print(f"Attempting to geocode address: {address}")
    
    # Method 1: Try with Nominatim
    try:
        location = geolocator.geocode(address, timeout=10, addressdetails=True)
        if location:
            print(f"Successfully geocoded with Nominatim: {location.address}")
            return location.latitude, location.longitude, location.address
    except Exception as e:
        print(f"Nominatim geocoding failed: {str(e)}")
    
    # Method 2: Try with Open-Meteo geocoding API
    try:
        # Extract city name from address
        parts = [p.strip() for p in address.split(',')]
        city = None
        state = None
        country = None
        
        # Try to identify components
        for part in reversed(parts):
            if not country and part.lower() in ['india', 'in']:
                country = 'India'
            elif not state and part.lower() in ['tamil nadu', 'karnataka', 'andhra pradesh', 'telangana', 'kerala', 'maharashtra', 'gujarat', 'rajasthan', 'punjab', 'haryana', 'uttar pradesh', 'madhya pradesh', 'bihar', 'west bengal', 'odisha', 'jharkhand', 'chhattisgarh', 'uttarakhand', 'himachal pradesh', 'jammu and kashmir', 'ladakh', 'delhi', 'goa', 'puducherry', 'chandigarh']:
                state = part
            elif not city and len(part) > 2:
                city = part
        
        if city:
            query = city
            if state:
                query += f", {state}"
            if country:
                query += f", {country}"
            
            print(f"Trying Open-Meteo with query: {query}")
            params = {
                "name": query,
                "count": 1,
                "language": "en",
                "format": "json"
            }
            
            response = requests.get(GEOCODING_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("results"):
                result = data["results"][0]
                lat = result["latitude"]
                lon = result["longitude"]
                
                # Try to get a more precise address using reverse geocoding
                try:
                    reverse_location = geolocator.reverse((lat, lon), timeout=10, addressdetails=True)
                    if reverse_location:
                        print(f"Reverse geocoded address: {reverse_location.address}")
                        return lat, lon, reverse_location.address
                except Exception as e:
                    print(f"Reverse geocoding failed: {str(e)}")
                
                # If reverse geocoding fails, use the Open-Meteo result
                name = result.get("name", "")
                admin1 = result.get("admin1", "")
                country_name = result.get("country", "")
                display = ", ".join([p for p in [name, admin1, country_name] if p])
                print(f"Successfully geocoded with Open-Meteo: {display}")
                return lat, lon, display
    except Exception as e:
        print(f"Open-Meteo geocoding failed: {str(e)}")
    
    # Method 3: Try with simplified address (just city and country)
    try:
        parts = [p.strip() for p in address.split(',')]
        if len(parts) >= 2:
            simplified = f"{parts[-2]}, {parts[-1]}"
            print(f"Trying simplified address: {simplified}")
            location = geolocator.geocode(simplified, timeout=10, addressdetails=True)
            if location:
                print(f"Successfully geocoded with simplified address: {location.address}")
                return location.latitude, location.longitude, location.address
    except Exception as e:
        print(f"Simplified address geocoding failed: {str(e)}")
    
    # Method 4: Try with just the city name
    try:
        parts = [p.strip() for p in address.split(',')]
        if len(parts) >= 2:
            city = parts[-2]
            print(f"Trying just city name: {city}")
            location = geolocator.geocode(city, timeout=10, addressdetails=True)
            if location:
                print(f"Successfully geocoded with city name: {location.address}")
                return location.latitude, location.longitude, location.address
    except Exception as e:
        print(f"City name geocoding failed: {str(e)}")
    
    # If all methods fail, raise an error
    raise ValueError(f"Could not geocode address: {address}. Please try a different address format.")

def fetch_historical_data(lat, lon):
    """Fetch historical weather data"""
    today = date.today()
    start_date = today - timedelta(days=365 * DAYS_HISTORY_YEARS)
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
    
    response = requests.get(HISTORICAL_URL, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    
    daily = data.get("daily") or {}
    times = daily.get("time") or []
    if not times:
        raise ValueError("No daily data returned")
    
    df = pd.DataFrame({
        "date": pd.to_datetime(times),
        "t_mean": daily.get("temperature_2m_mean"),
        "t_max": daily.get("temperature_2m_max"),
        "t_min": daily.get("temperature_2m_min"),
        "precip": daily.get("precipitation_sum"),
        "wind_max": daily.get("wind_speed_10m_max"),
        "rh_mean": daily.get("relative_humidity_2m_mean"),
    })
    
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["t_mean"])
    return df

def build_training_data(df):
    """Build training data for ML models"""
    df = df.copy()
    df["doy"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.0)
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
    
    # Deltas to reconstruct t_min/t_max
    delta_max = (df_train["t_max"] - df_train["t_mean"]).dropna().mean()
    delta_min = (df_train["t_mean"] - df_train["t_min"]).dropna().mean()
    
    if math.isnan(delta_max):
        delta_max = 2.0
    if math.isnan(delta_min):
        delta_min = 2.0
    
    return X_reg, y_reg, X_cls, y_cls, delta_max, delta_min, df, df_train

def train_models(X_reg, y_reg, X_cls, y_cls):
    """Train ML models"""
    # Train regression model
    reg_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg_model.fit(X_reg, y_reg)
    
    # Train classification model
    cls_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    cls_model.fit(X_cls, y_cls)
    
    return reg_model, cls_model

def forecast_next_days(df, reg_model, cls_model, delta_max, delta_min, n_days):
    """Forecast weather for next n_days"""
    df = df.sort_values("date")
    last_row = df.iloc[-1]
    last_hist_t_mean = float(last_row["t_mean"])
    
    today = date.today()
    current_date = today - timedelta(days=1)
    current_t_mean = last_hist_t_mean
    
    forecasts = []
    
    for _ in range(n_days):
        forecast_date = current_date + timedelta(days=1)
        doy = forecast_date.timetuple().tm_yday
        sin_doy = math.sin(2 * math.pi * doy / 365.0)
        cos_doy = math.cos(2 * math.pi * doy / 365.0)
        
        # Regression: predict next mean temp
        X_reg_new = np.array([[doy, sin_doy, cos_doy, current_t_mean]])
        t_mean_next = float(reg_model.predict(X_reg_new)[0])
        
        # Classification: rain probability
        X_cls_new = np.array([[doy, sin_doy, cos_doy]])
        proba = cls_model.predict_proba(X_cls_new)[0]
        prob_rain = float(proba[1])
        rain_yes = prob_rain >= 0.5
        
        # Approximate min/max around mean
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
        
        current_date = forecast_date
        current_t_mean = t_mean_next
    
    return forecasts

def generate_forecast_table(forecast_list, out_png):
    """Generate forecast table as PNG"""
    headers = ["Date", "Tmean(°C)", "Tmax", "Tmin", "RainProb(%)", "Rain?"]
    
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

def generate_pdf_report(location_display, lat, lon, forecast_list, png_table, out_pdf):
    """Generate PDF report"""
    doc = SimpleDocTemplate(out_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = styles["Title"]
    normal_style = styles["Normal"]
    
    story.append(Paragraph("Weather Forecast Report", title_style))
    story.append(Spacer(1, 0.2 * inch))
    
    info_text = f"Location: {location_display} (lat={lat:.4f}, lon={lon:.4f})"
    story.append(Paragraph(info_text, normal_style))
    story.append(Spacer(1, 0.1 * inch))
    
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")
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

def get_weather_forecast(address, forecast_days):
    """Main function to get weather forecast"""
    # Geocode address
    lat, lon, display = geocode_address(address)
    
    # Fetch historical data
    df_hist = fetch_historical_data(lat, lon)
    
    # Build training data
    X_reg, y_reg, X_cls, y_cls, delta_max, delta_min, df_processed, df_train = build_training_data(df_hist)
    
    # Train models
    reg_model, cls_model = train_models(X_reg, y_reg, X_cls, y_cls)
    
    # Forecast
    forecast_list = forecast_next_days(df_processed, reg_model, cls_model, delta_max, delta_min, forecast_days)
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    png_path = f"static/reports/forecast_{unique_id}.png"
    pdf_path = f"static/reports/report_{unique_id}.pdf"
    
    # Generate table and report
    generate_forecast_table(forecast_list, png_path)
    generate_pdf_report(display, lat, lon, forecast_list, png_path, pdf_path)
    
    return forecast_list, pdf_path, png_path