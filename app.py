import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "randomforestregg.pkl")
    return joblib.load(model_path)

model = load_model()

st.title("Amazon Delivery Time Prediction")
st.success("Model loaded successfully!")
# -------------------------------
# User Inputs
# -------------------------------
agent_age = st.number_input("Agent Age", 18, 65, 30)
agent_rating = st.number_input("Agent Rating", 1.0, 5.0, 4.0)

store_lat = st.number_input("Store Latitude", value=12.97)
store_lon = st.number_input("Store Longitude", value=77.59)
drop_lat = st.number_input("Drop Latitude", value=12.93)
drop_lon = st.number_input("Drop Longitude", value=77.62)

order_time = st.time_input("Order Time", value=datetime.now().time())
order_date = st.date_input("Order Date")

weather = st.selectbox("Weather", ["Sunny", "Rainy", "Foggy", "Stormy"])
traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
vehicle = st.selectbox("Vehicle", ["Bike", "Scooter", "Car", "Van"])
area = st.selectbox("Area", ["Urban", "Metropolitan", "Semi-Urban"])
category = st.selectbox("Product Category", ["Electronics", "Grocery", "Clothing", "Food"])

# -------------------------------
# Feature Engineering (SAME AS TRAINING)
# -------------------------------
distance_km = ((store_lat - drop_lat)**2 + (store_lon - drop_lon)**2) ** 0.5 * 111

order_hour = order_time.hour
is_weekday = order_date.weekday() < 5  # Mon–Fri = True

# -------------------------------
# Create input dataframe
# -------------------------------
input_df = pd.DataFrame({
    "Agent_Age": [agent_age],
    "Agent_Rating": [agent_rating],
    "distance_km": [distance_km],
    "order_hour": [order_hour],
    "is_weekday": [int(is_weekday)],
    "Weather": [weather],
    "Traffic": [traffic],
    "Vehicle": [vehicle],
    "Area": [area],
    "Category": [category]
})

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Delivery Time"):
    prediction = model.predict(input_df)
    st.success(f"⏱ Estimated Delivery Time: {prediction[0]:.2f} hours")


    
