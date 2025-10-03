# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime
import os

st.set_page_config(page_title="Weather Reporter", layout="wide")

# -------------------------
# Helpers / Config
# -------------------------
@st.cache_data
def load_historical(csv_path: str = "expanded_filled_weather_data.csv"):
    df = pd.read_csv(csv_path)
    # Basic parse for DATE if necessary
    if "DATE" in df.columns:
        try:
            df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m")
        except Exception:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    return df

def get_openweather_key():
    # prefer streamlit secrets
    try:
        return st.secrets["openweather"]["api_key"]
    except Exception:
        return os.getenv("OPENWEATHER_API_KEY", "")

def fetch_current_weather_by_city(city: str, apikey: str):
    """Fetch current weather using OpenWeatherMap Current Weather API (city name)."""
    base = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": apikey, "units": "metric"}
    resp = requests.get(base, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def fetch_current_weather_by_coords(lat: float, lon: float, apikey: str):
    base = "https://api.openweathermap.org/data/2.5/onecall"
    params = {"lat": lat, "lon": lon, "appid": apikey, "units": "metric", "exclude": "minutely,hourly,alerts"}
    resp = requests.get(base, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

# -------------------------
# Prediction model utilities
# -------------------------
MODEL_PATH = "temp_model.joblib"

def train_simple_model(df: pd.DataFrame, model_path: str = MODEL_PATH):
    """
    Train a simple LinearRegression to predict TAVG from month, lat, lon, elevation, and TMIN/TMAX if available.
    This is intentionally simple so it trains quickly in-app for demonstration.
    """
    # Prepare features
    df2 = df.copy().dropna(subset=["TAVG"])
    # Extract numeric month
    if "DATE" in df2.columns:
        df2["month"] = pd.to_datetime(df2["DATE"], errors="coerce").dt.month.fillna(0).astype(int)
    else:
        df2["month"] = 0

    features = []
    for col in ["LATITUDE", "LONGITUDE", "ELEVATION", "month"]:
        if col in df2.columns:
            features.append(col)

    # Add TMIN/TMAX if present
    for col in ["TMIN", "TMAX"]:
        if col in df2.columns:
            features.append(col)

    if len(features) < 1:
        raise RuntimeError("Not enough features to train.")

    X = df2[features].fillna(0).astype(float)
    y = df2["TAVG"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Save model + feature list
    joblib.dump({"model": model, "features": features}, model_path)
    return model, features

def load_or_train_model(df: pd.DataFrame):
    if os.path.exists(MODEL_PATH):
        obj = joblib.load(MODEL_PATH)
        return obj["model"], obj["features"]
    else:
        model, features = train_simple_model(df)
        joblib.dump({"model": model, "features": features}, MODEL_PATH)
        return model, features

def predict_temp(model, features, input_dict):
    X = pd.DataFrame([{k: input_dict.get(k, 0) for k in features}])
    return float(model.predict(X)[0])

# -------------------------
# UI
# -------------------------
st.title("🌤️ Weather Reporter — Streamlit App")
st.markdown("Upload historical CSV (optional), fetch live weather, and predict temperature using a simple model.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data")
    uploaded = st.file_uploader("Upload historical CSV (optional)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        # fallback to file provided in project folder
        if os.path.exists("expanded_filled_weather_data.csv"):
            df = load_historical("expanded_filled_weather_data.csv")
            st.success("Loaded local CSV: expanded_filled_weather_data.csv")
        else:
            df = pd.DataFrame()
            st.info("No CSV loaded yet. Upload one or add expanded_filled_weather_data.csv to project folder.")

    if not df.empty:
        st.write("Data preview")
        st.dataframe(df.head(10))
        st.write("Columns:", list(df.columns))

with col2:
    st.header("Live Weather")
    apikey = get_openweather_key()
    if not apikey:
        st.warning("No OpenWeather API key found. Add it to .streamlit/secrets.toml or set OPENWEATHER_API_KEY env var.")
    city = st.text_input("Enter city name (e.g., Delhi, Kolkata)", value="New Delhi")
    lat = st.number_input("Latitude (optional)", value=0.0, format="%.6f")
    lon = st.number_input("Longitude (optional)", value=0.0, format="%.6f")

    if st.button("Fetch current weather"):
        try:
            if city and apikey:
                data = fetch_current_weather_by_city(city, apikey)
                st.subheader(f"Current weather in {city}")
                st.json(data)
                # show simple text
                main = data.get("main", {})
                st.metric("Temperature (°C)", main.get("temp"))
                st.metric("Feels like (°C)", main.get("feels_like"))
            elif lat != 0.0 and lon != 0.0 and apikey:
                data = fetch_current_weather_by_coords(lat, lon, apikey)
                st.subheader(f"Current weather at {lat},{lon}")
                st.json(data)
                st.metric("Current temp (°C)", data["current"].get("temp"))
            else:
                st.error("Provide a city name or lat/lon and configure API key.")
        except Exception as e:
            st.error(f"Error fetching weather: {e}")

# -------------------------
# Train / Predict
# -------------------------
st.header("Train / Predict (simple model)")

if st.button("Train model (quick)"):
    if df.empty:
        st.error("No historical CSV/data available to train on.")
    else:
        with st.spinner("Training model..."):
            try:
                model, features = train_simple_model(df)
                st.success("Model trained and saved to disk.")
                st.write("Features used:", features)
            except Exception as e:
                st.error(f"Training failed: {e}")

# Predict UI
st.subheader("Predict TAVG (by input values)")
# Build input form dynamically from features if model exists
if os.path.exists(MODEL_PATH):
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]
    st.write("Model features:", features)
    input_vals = {}
    for f in features:
        if f in ["LATITUDE", "LONGITUDE"]:
            input_vals[f] = st.number_input(f, value=0.0, format="%.6f")
        elif f == "month":
            input_vals[f] = st.selectbox("month", list(range(1,13)), index=datetime.now().month-1)
        else:
            input_vals[f] = st.number_input(f, value=0.0)
    if st.button("Predict"):
        pred = predict_temp(model, features, input_vals)
        st.metric("Predicted TAVG (°C)", f"{pred:.2f}")
else:
    st.info("No saved model found. Click 'Train model (quick)' to create one from your CSV.")

# -------------------------
# Simple visualization
# -------------------------
# ✅ Safe aggregation for monthly averages
# Make sure we have data to plot
if 'DATE' in df.columns:
    df_plot = df.copy()
    df_plot['DATE'] = pd.to_datetime(df_plot['DATE'], errors='coerce')  # ensure DATE is datetime
else:
    df_plot = pd.DataFrame()
    
        # Convert Period back to datetime for plotting
    agg["DATE"] = agg["DATE"].dt.to_timestamp()

    st.subheader("📊 Monthly Average Weather Trends")
    st.line_chart(agg.set_index("DATE"))  # plots all numeric cols

except Exception as e:
    st.error(f"Plot failed: {e}")
else:
    st.warning("DATE column not found in dataset.")







