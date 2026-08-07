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

# Browser left side Sidebar
st.sidebar.title("Project Details")

st.sidebar.write("Project : AI Based Accident Risk Prediction System")

st.sidebar.write("Algorithm : Random Forest")

st.sidebar.write("Language : Python")

st.sidebar.write("Framework : Streamlit")

# Title
st.title("🚗 AI Based Accident Risk Prediction System")

# Project Description
st.write("""
This AI system predicts accident risk using Machine Learning.
It analyzes Weather, Vehicle Speed and Traffic conditions
to identify accident risk.
""")

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
    # Risk Percentage
    if risk == "High":
        st.progress(95)
        
        st.write("Risk Probability : 95%")
    
    elif risk == "Medium":
        
        st.progress(60)
        
        st.write("Risk Probability : 60%")
    
    else:
        
        st.progress(20)
        
        st.write("Risk Probability : 20%")
    
    # Safety Tips
        if risk == "High":
            st.error("🚨 High Accident Risk")
            
            st.subheader("Safety Tips")
            
            st.write("✔ Reduce Vehicle Speed")
            st.write("✔ Wear Seat Belt")
            st.write("✔ Maintain Safe Distance")
            st.write("✔ Avoid Mobile Phone While Driving")
        elif risk == "Medium":
            st.warning("⚠️ Medium Accident Risk")
            
            st.subheader("Safety Tips")
            
            st.write("✔ Drive Carefully")
            st.write("✔ Follow Traffic Rules")
        
        else:
          
            st.success("✅ Low Accident Risk")
            
            st.subheader("Safety Tips")
            
            st.write("✔ Continue Safe Driving")


#Footer
st.markdown("---")

st.write("Developed By : Sudharsan V")

st.write("Department : AI & DS")
