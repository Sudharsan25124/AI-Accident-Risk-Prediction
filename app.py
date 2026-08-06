import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("accident_data.csv")

# Encode input columns
weather_encoder = LabelEncoder()
traffic_encoder = LabelEncoder()
risk_encoder = LabelEncoder()

data["Weather"] = weather_encoder.fit_transform(data["Weather"])
data["Traffic"] = traffic_encoder.fit_transform(data["Traffic"])
data["Risk"] = risk_encoder.fit_transform(data["Risk"])

# Load trained model
model = joblib.load("accident_model.pkl")

# Title
st.title("🚗 AI Based Accident Risk Prediction System")

# Inputs
weather = st.selectbox("Weather", weather_encoder.classes_)
speed = st.slider("Vehicle Speed", 0, 150, 50)
traffic = st.selectbox("Traffic", traffic_encoder.classes_)

# Prediction
if st.button("Predict"):

    weather_value = weather_encoder.transform([weather])[0]
    traffic_value = traffic_encoder.transform([traffic])[0]

    prediction = model.predict([[weather_value, speed, traffic_value]])

    risk = risk_encoder.inverse_transform(prediction)[0]

    if risk == "High":
        st.error("🚨 High Accident Risk")

    elif risk == "Medium":
        st.warning("⚠️ Medium Accident Risk")

    else:
        st.success("✅ Low Accident Risk")