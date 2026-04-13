import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model("model.h5")

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


st.title("🌱 Irrigation Prediction App")
st.write("Enter values to predict irrigation need")


temperature = st.number_input("Temperature (°C)")
soil_moisture = st.number_input("Soil Moisture")
rainfall = st.number_input("Rainfall (mm)")
sunlight = st.number_input("Sunlight Hours")
previous_irrigation = st.number_input("Previous Irrigation (mm)")


water_stress = temperature / (soil_moisture + 1)
evaporation = temperature * sunlight
total_water = rainfall + previous_irrigation


if st.button("Predict"):

    input_dict = {
        # your inputs
        "Temperature_C": temperature,
        "Soil_Moisture": soil_moisture,
        "Rainfall_mm": rainfall,
        "Sunlight_Hours": sunlight,
        "Previous_Irrigation_mm": previous_irrigation,

        # engineered
        "water_stress": water_stress,
        "evaporation": evaporation,
        "total_water": total_water,


        "Soil_Type": 0,
        "Soil_pH": 0,
        "Organic_Carbon": 0,
        "Electrical_Conductivity": 0,
        "Humidity": 0,
        "Wind_Speed_kmh": 0,
        "Crop_Type": 0,
        "Crop_Growth_Stage": 0,
        "Season": 0,
        "Field_Area_hectare": 0,
        "Mulching_Used": 0,
        "Region": 0,

        "evapotranspiration_proxy": 0,
        "rain_balance": 0,
        "moisture_temp": 0,
        "humidity_temp": 0,
        "humidity_per_moisture": 0,
        "temp_per_conductivity": 0,
        "temp_humidity_ratio": 0,
        "moisture_gap": 0,
        "rain_effect": 0,
        "effective_water_input": 0,
        "moisture_deficit": 0,
        "temp_moisture_ratio": 0,
    }

  
    input_data = pd.DataFrame([input_dict])

 
    input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)


    input_scaled = scaler.transform(input_data)

  
    prediction = model.predict(input_scaled)
    result = np.argmax(prediction)

    labels = ["Low", "Medium", "High"]

    st.success(f"Irrigation Need: {labels[result]}")