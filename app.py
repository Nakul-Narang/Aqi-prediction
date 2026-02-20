import streamlit as st
import pandas as pd
import joblib
import streamlit as st

st.write("App is running")
st.set_page_config(page_title="AQI Prediction App")

st.title("🌫 Air Pollution Prediction System")

# Load model and encoders
model = joblib.load("pollution_model.pkl")
city_encoder = joblib.load("city_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")
pollution_encoder = joblib.load("pollution_level_encoder.pkl")

st.write("Select details to predict pollution level")

city = st.selectbox("Select City", city_encoder.classes_)
state = st.selectbox("Select State", state_encoder.classes_)
pollution_type = st.selectbox("Pollution Type", pollution_encoder.classes_)

year = st.number_input("Year", 2015, 2035, 2024)
month = st.number_input("Month", 1, 12, 1)
day = st.number_input("Day", 1, 31, 1)

if st.button("Predict Pollution Level"):
    city_enc = city_encoder.transform([city])[0]
    state_enc = state_encoder.transform([state])[0]
    pollution_enc = pollution_encoder.transform([pollution_type])[0]

    input_df = pd.DataFrame({
    'city_enc': [city_enc],
    'state_enc': [state_enc],
    'pollutant_enc': [pollution_enc],  # FIXED NAME
    'year': [year],
    'month': [month],
    'day': [day]
})

    prediction = model.predict(input_df)

    st.success(f"Predicted Pollution Level: {prediction[0]:.2f}")
