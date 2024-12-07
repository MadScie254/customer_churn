import streamlit as st
import pandas as pd
import joblib

# Load the saved XGBoost model
model = joblib.load('xgb_churn_model.pkl')

# Define the column names (feature names) expected by the model
model_columns = [
    'Call Failure', 'Complains', 'Subscription Length', 'Charge Amount', 
    'Seconds of Use', 'Frequency of use', 'Frequency of SMS', 
    'Distinct Called Numbers', 'Age Group', 'Tariff Plan', 
    'Status', 'Age', 'Customer Value'
]

# Define default values for features not provided by the user
default_values = {
    'Call Failure': 0,  # No call failure by default
    'Complains': 0,      # No complaints by default
    'Seconds of Use': 4472,  # Average seconds of use
    'Frequency of use': 69,  # Average frequency of use
    'Frequency of SMS': 73,  # Average frequency of SMS
    'Distinct Called Numbers': 23,  # Average distinct called numbers
    'Age Group': 2,  # Default middle age group
    'Tariff Plan': 1,  # Default tariff plan
    'Status': 1,  # Active status by default
    'Customer Value': 470.97  # Default customer value
}

# Function to predict churn
def predict_churn(input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit app layout
st.title("Customer Churn Prediction")
st.write("This app predicts if a customer will churn based on the input features.")

# Collect user input for essential features
age = st.slider('Age', 15, 55, 30)
subscription_length = st.slider('Subscription Length', 3, 47, 35)
charge_amount = st.slider('Charge Amount', 0, 10, 1)

# Prepare the input data (using default values for non-user features)
input_data = pd.DataFrame({
    'Call Failure': [default_values['Call Failure']],
    'Complains': [default_values['Complains']],
    'Subscription Length': [subscription_length],
    'Charge Amount': [charge_amount],
    'Seconds of Use': [default_values['Seconds of Use']],
    'Frequency of use': [default_values['Frequency of use']],
    'Frequency of SMS': [default_values['Frequency of SMS']],
    'Distinct Called Numbers': [default_values['Distinct Called Numbers']],
    'Age Group': [default_values['Age Group']],
    'Tariff Plan': [default_values['Tariff Plan']],
    'Status': [default_values['Status']],
    'Age': [age],
    'Customer Value': [default_values['Customer Value']]
})

# Reorder the columns to match the model's expected order
input_data = input_data[model_columns]

# Get the prediction
prediction = predict_churn(input_data)

# Display the prediction
if prediction == 1:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")
