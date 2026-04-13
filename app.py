import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# ----------------------------
# Load Model & Preprocessing
# ----------------------------
model = tf.keras.models.load_model("model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ----------------------------
# App UI
# ----------------------------
st.title("🌱 Irrigation Need Prediction")

st.write("Enter environmental data to predict irrigation level")

# Inputs
temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
soil_moisture = st.number_input("Soil Moisture", 0.0, 100.0, 50.0)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 10.0)
sunlight = st.number_input("Sunlight Hours", 0.0, 24.0, 8.0)
previous_irrigation = st.number_input("Previous Irrigation", 0.0, 100.0, 10.0)

# Feature Engineering (same as notebook)
water_stress = temperature / (soil_moisture + 1)
evaporation = temperature * sunlight
total_water = rainfall + previous_irrigation

# Prepare input
input_data = pd.DataFrame([{
    "Temperature_C": temperature,
    "Soil_Moisture": soil_moisture,
    "Rainfall_mm": rainfall,
    "Sunlight_Hours": sunlight,
    "Previous_Irrigation": previous_irrigation,
    "water_stress": water_stress,
    "evaporation": evaporation,
    "total_water": total_water
}])

# Scale
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    class_idx = np.argmax(prediction)

    labels = ["Low", "Medium", "High"]
    result = labels[class_idx]

    st.success(f"💧 Irrigation Need: {result}")
