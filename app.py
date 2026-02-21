import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("pollution_model.pkl")
city_encoder = joblib.load("city_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")
pollution_encoder = joblib.load("pollution_level_encoder.pkl")

st.title("🌫 Air Pollution Prediction System")

city = st.selectbox("Select City", city_encoder.classes_)
state = st.selectbox("Select State", state_encoder.classes_)
pollution_type = st.selectbox("Pollution Type", pollution_encoder.classes_)

year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
day = st.number_input("Day", min_value=1, max_value=31, value=1)

if st.button("Predict"):

    # Encode values
    city_enc = city_encoder.transform([city])[0]
    state_enc = state_encoder.transform([state])[0]
    pollutant_enc = pollution_encoder.transform([pollution_type])[0]

    # Create DataFrame EXACTLY matching training
    input_df = pd.DataFrame({
        'city_enc': [city_enc],
        'state_enc': [state_enc],
        'pollutant_enc': [pollutant_enc],
        'year': [year],
        'month': [month],
        'day': [day]
    })

    # Ensure correct column order
    input_df = input_df[['city_enc', 'state_enc', 'pollutant_enc', 'year', 'month', 'day']]

    prediction = model.predict(input_df)

    st.success(f"Predicted AQI Value: {prediction[0]}")
