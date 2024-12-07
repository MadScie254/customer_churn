import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved XGBoost model
model = joblib.load('xgb_churn_model.pkl')

# Extract feature names from the model
model_features = model.get_booster().feature_names

# Function to align input features dynamically
def align_features(input_data):
    # Ensure correct column order by reordering to match model_features
    input_data = input_data.reindex(columns=model_features, fill_value=0)
    return input_data

# Streamlit app layout
st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a telecom customer is likely to churn.")

# Collect user inputs interactively
st.sidebar.header("Input Customer Details:")
age = st.sidebar.number_input("Age", min_value=15, max_value=55, value=30)
subscription_length = st.sidebar.number_input("Subscription Length (months)", min_value=3, max_value=47, value=35)
charge_amount = st.sidebar.slider("Charge Amount", 0.0, 10.0, 1.0)
seconds_of_use = st.sidebar.number_input("Seconds of Use", 0, 20000, 5000)
frequency_of_use = st.sidebar.number_input("Frequency of Use", 0, 300, 50)
frequency_of_sms = st.sidebar.number_input("Frequency of SMS", 0, 600, 100)
distinct_called_numbers = st.sidebar.number_input("Distinct Called Numbers", 0, 100, 20)
age_group = st.sidebar.selectbox("Age Group", [1, 2, 3, 4, 5])
tariff_plan = st.sidebar.selectbox("Tariff Plan", [1, 2])
status = st.sidebar.selectbox("Status", [1, 2])

# Create input DataFrame
input_data = pd.DataFrame({
    'Call Failure': [0],  # Default values
    'Complains': [0],     # Default values
    'Subscription Length': [subscription_length],
    'Charge Amount': [charge_amount],
    'Seconds of Use': [seconds_of_use],
    'Frequency of use': [frequency_of_use],
    'Frequency of SMS': [frequency_of_sms],
    'Distinct Called Numbers': [distinct_called_numbers],
    'Age Group': [age_group],
    'Tariff Plan': [tariff_plan],
    'Status': [status],
    'Age': [age],
    'Customer Value': [0]  # Default values
})

# Align features dynamically
input_data = align_features(input_data)

# Predict churn
if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is unlikely to churn.")
    except ValueError as e:
        st.error(f"Prediction Error: {e}")
