
-# 🌦️ Weather Data App
+# 🌦️ Weather Reporter App
 
-This is a basic Streamlit application that displays weather , predicting temperature,humidity for actual years .data collected from a CSV file and visualizes numeric columns using line charts.
+A Streamlit application to explore historical weather data, fetch live weather details from OpenWeather, and predict average temperature (`TAVG`) using a lightweight linear regression model.
 
 ## 🚀 Features
-- Displays dataset preview
-- Automatically plots numeric columns
-- Built using Streamlit and Pandas
+- Upload a custom historical CSV or use the bundled `expanded_filled_weather_data.csv`
+- Preview data and inspect available columns
+- Fetch current weather by city or coordinates (with OpenWeather API key)
+- Train and save a quick regression model in-app (`temp_model.joblib`)
+- Predict `TAVG` from model features
+- Visualize monthly average temperature trends and numeric data series
 
 ## 📁 Project Structure
-├── app.py
-├── Weather.csv
-├── requirements.txt
-└── README.md
+- `app.py` — Main Streamlit app
+- `expanded_filled_weather_data.csv` — Sample historical dataset
+- `requirements.txt` — Python dependencies
+- `README.md` — Project documentation
 
 ## ▶️ Run Locally
 ```bash
 pip install -r requirements.txt
 streamlit run app.py
+```
+
+## 🔐 API Key Setup (OpenWeather)
+You can provide your OpenWeather API key in either of these ways:
+
+1. Environment variable:
+```bash
+export OPENWEATHER_API_KEY="your_api_key_here"
+```
+
+2. Streamlit secrets (`.streamlit/secrets.toml`):
+```toml
+[openweather]
+api_key = "your_api_key_here"
+```
 
EOF
)
