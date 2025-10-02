{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "edd47108-6b6b-4701-b1c2-1473aec7d537",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import requests\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.linear_model import LinearRegression\n",
    "import numpy as np\n",
    "\n",
    "# OpenWeather API details\n",
    "API_KEY = \"57cd13ca4adb8ba03be5e0925596fac5\"\n",
    "BASE_URL = \"http://api.openweathermap.org/data/2.5/weather?\"\n",
    "\n",
    "st.title(\"🌦️ Weather Data Analysis & Prediction App\")\n",
    "\n",
    "# User input\n",
    "city = st.text_input(\"Enter City Name\", \"Visakhapatnam\")\n",
    "\n",
    "if st.button(\"Get Weather\"):\n",
    "    # Fetch real-time weather\n",
    "    url = BASE_URL + \"q=\" + city + \"&appid=\" + API_KEY + \"&units=metric\"\n",
    "    response = requests.get(url).json()\n",
    "\n",
    "    if response[\"cod\"] != \"404\":\n",
    "        main = response[\"main\"]\n",
    "        temp = main[\"temp\"]\n",
    "        humidity = main[\"humidity\"]\n",
    "        pressure = main[\"pressure\"]\n",
    "        weather_desc = response[\"weather\"][0][\"description\"]\n",
    "\n",
    "        st.success(f\"📍 {city}\")\n",
    "        st.write(f\"🌡️ Temperature: {temp}°C\")\n",
    "        st.write(f\"💧 Humidity: {humidity}%\")\n",
    "        st.write(f\"🔽 Pressure: {pressure} hPa\")\n",
    "        st.write(f\"🌥️ Weather: {weather_desc}\")\n",
    "\n",
    "        # Dummy dataset for prediction (replace with your real model/data)\n",
    "        X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)   # Days\n",
    "        y = np.array([temp-2, temp-1, temp, temp+1, temp+2])  # Temperatures\n",
    "\n",
    "        model = LinearRegression()\n",
    "        model.fit(X, y)\n",
    "\n",
    "        future_days = np.array([6, 7, 8]).reshape(-1, 1)\n",
    "        future_preds = model.predict(future_days)\n",
    "\n",
    "        st.subheader(\"📈 Predicted Temperature (Next Days)\")\n",
    "        fig, ax = plt.subplots()\n",
    "        ax.plot(X, y, marker=\"o\", label=\"Past Temp\")\n",
    "        ax.plot(future_days, future_preds, marker=\"x\", linestyle=\"--\", label=\"Predicted Temp\")\n",
    "        ax.set_xlabel(\"Days\")\n",
    "        ax.set_ylabel(\"Temperature (°C)\")\n",
    "        ax.legend()\n",
    "        st.pyplot(fig)\n",
    "\n",
    "    else:\n",
    "        st.error(\"City Not Found ❌\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
