# app.py

import streamlit as st
import pandas as pd
import joblib
import os

from src.data_preprocessing import encode_categorical, feature_engineering
from src.models import train_random_forest, evaluate_model
from src.recommendations import generate_health_advice

# Load pre-trained model and scaler
MODEL_DIR = os.path.join("outputs", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "final_health_risk_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.warning("Model artifacts not found under outputs/models/. The app will still render inputs.")
    model = None
    scaler = None
else:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

st.title("☕ Predictive Health Risk App")
st.write("Predict your health risk based on lifestyle factors!")

# Sidebar input
st.sidebar.header("Input Your Details")
age = st.sidebar.slider("Age", 18, 80, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
coffee_intake = st.sidebar.slider("Coffee Intake (cups/day)", 0, 10, 2)
caffeine_mg = st.sidebar.number_input("Caffeine mg/day", 0, 1000, 100)
sleep_hours = st.sidebar.slider("Sleep Hours", 0, 12, 7)
sleep_quality = st.sidebar.selectbox("Sleep Quality", ["Poor", "Average", "Good"])
bmi = st.sidebar.number_input("BMI", 10, 50, 22)
heart_rate = st.sidebar.number_input("Heart Rate", 50, 120, 70)
stress_level = st.sidebar.slider("Stress Level (1-10)", 0, 10, 5)
physical_activity = st.sidebar.slider("Physical Activity Hours/week", 0, 20, 3)
smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
alcohol = st.sidebar.selectbox("Alcohol Consumption", ["No", "Yes"])

# Collect input into DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Coffee_Intake": coffee_intake,
    "Caffeine_mg": caffeine_mg,
    "Sleep_Hours": sleep_hours,
    "Sleep_Quality": sleep_quality,
    "BMI": bmi,
    "Heart_Rate": heart_rate,
    "Stress_Level": stress_level,
    "Physical_Activity_Hours": physical_activity,
    "Smoking": smoking,
    "Alcohol_Consumption": alcohol,
    # Add placeholders for missing features
    "Country": "Unknown",
    "Occupation": "Other"
}])

# Preprocess input
input_data = encode_categorical(input_data)
input_data = feature_engineering(input_data)

# Ensure column order matches scaler
scaled_input = None
if scaler is not None:
    input_data = input_data[scaler.feature_names_in_]  # reorder columns
    scaled_input = scaler.transform(input_data)

# Prediction
if st.button("Predict Health Risk"):
    if model is None or scaled_input is None:
        st.error("Model/scaler not available. Please train and save artifacts to outputs/models/.")
    else:
        risk = model.predict(scaled_input)[0]
        st.subheader(f"Predicted Health Risk: {risk}")
    
    # Generate advice
    advice = generate_health_advice(input_data.iloc[0].to_dict())
    st.subheader("Recommended Actions:")
    for a in advice:
        st.write(f"- {a}")
