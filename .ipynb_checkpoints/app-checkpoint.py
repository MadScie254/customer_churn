import streamlit as st
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import requests

# Load the saved XGBoost model
model = joblib.load('xgb_churn_model.pkl')

# Extract feature names from the model
model_features = model.get_booster().feature_names

# Function to align input features dynamically
def align_features(input_data):
    input_data = input_data.reindex(columns=model_features, fill_value=0)
    return input_data

# Function to load Lottie animations
def load_lottie_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# Load animations
loading_animation = load_lottie_url("https://assets8.lottiefiles.com/private_files/lf30_5ttqPi.json")
churn_animation = load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_eeuhmcne.json")
success_animation = load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_oftaxdnf.json")

# Streamlit Page Configuration
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# Header Section
st.markdown(
    """
    <div style="background-color:#006d77;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h1 style="color:white;text-align:center;">📊 Telecom Customer Churn Prediction</h1>
        <p style="color:white;text-align:center;">An engaging app to predict customer churn with dynamic animations and live feedback!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for Inputs
st.sidebar.header("Customer Data Input")
st.sidebar.markdown(
    """
    <div style="background-color:#e3f2fd;padding:10px;border-radius:10px;">
        <p style="text-align:center;font-size:14px;">Adjust the sliders and dropdowns to input customer data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input Widgets
age = st.sidebar.number_input("Age", min_value=15, max_value=55, value=30)
subscription_length = st.sidebar.number_input("Subscription Length (months)", min_value=3, max_value=47, value=24)
charge_amount = st.sidebar.slider("Charge Amount", 0.0, 10.0, 5.0, step=0.1)
seconds_of_use = st.sidebar.number_input("Seconds of Use", 0, 20000, 5000)
frequency_of_use = st.sidebar.number_input("Frequency of Use", 0, 300, 50)
frequency_of_sms = st.sidebar.number_input("Frequency of SMS", 0, 600, 100)
distinct_called_numbers = st.sidebar.number_input("Distinct Called Numbers", 0, 100, 20)
age_group = st.sidebar.selectbox("Age Group", [1, 2, 3, 4, 5])
tariff_plan = st.sidebar.selectbox("Tariff Plan", [1, 2])
status = st.sidebar.selectbox("Status", [1, 2])

# DataFrame for input
input_data = pd.DataFrame({
    'Call Failure': [0],
    'Complains': [0],
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
    'Customer Value': [0]
})

# Align Features
input_data = align_features(input_data)

# Main Layout with Animations
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Predict Churn")
    st_lottie(loading_animation, height=200, key="loading")

    if st.button("🔮 Predict Now"):
        with st.spinner("Analyzing customer data..."):
            try:
                prediction = model.predict(input_data)
                if prediction[0] == 1:
                    st.markdown(
                        """
                        <div style="background-color:#f94144;padding:20px;border-radius:10px;text-align:center;">
                            <h2 style="color:white;">⚠️ Prediction: Likely to Churn</h2>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st_lottie(churn_animation, height=200, key="churn")
                else:
                    st.markdown(
                        """
                        <div style="background-color:#2a9d8f;padding:20px;border-radius:10px;text-align:center;">
                            <h2 style="color:white;">✅ Prediction: Unlikely to Churn</h2>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st_lottie(success_animation, height=200, key="success")
            except Exception as e:
                st.error(f"Prediction Error: {e}")

with col2:
    st.markdown("### About the Model")
    st.markdown(
        """
        - **Model Algorithm**: XGBoost Classifier
        - **Accuracy**: 94%
        - **ROC-AUC Score**: 0.88
        
        #### Features Used:
        - Subscription Length
        - Charge Amount
        - Seconds of Use
        - Frequency of Use & SMS
        - Distinct Called Numbers
        - Customer Demographics

        #### How to Use:
        1. Input customer data on the left sidebar.
        2. Click **Predict Now** for instant churn analysis.
        """
    )

# Footer
st.markdown(
    """
    <div style="background-color:#006d77;padding:15px;border-radius:10px;margin-top:20px;">
        <p style="color:white;text-align:center;">Developed with ❤️ by Daniel Wanjala. Powered by Streamlit and XGBoost.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
