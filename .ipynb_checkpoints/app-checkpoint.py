import streamlit as st
import pandas as pd
import joblib
import requests
import logging
from streamlit_lottie import st_lottie

# =================== PAGE CONFIGURATION ===================
st.set_page_config(
    page_title="Telecom Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================== CONFIGURATION & SETUP ===================
logging.basicConfig(filename='app.log', level=logging.ERROR)

# =================== MODEL LOADING ===================
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgb_churn_model.pkl')
    except Exception as e:
        logging.error(f"Model loading error: {str(e)}")
        st.error("Failed to load the prediction model. Please check the model file.")
        st.stop()

model = load_model()
model_features = model.get_booster().feature_names

# =================== HELPER FUNCTIONS ===================
def align_features(input_data):
    """Ensure input data matches model features"""
    try:
        input_data = input_data.reindex(columns=model_features, fill_value=0)
        if not all(input_data.columns == model_features):
            raise ValueError("Feature alignment failed")
        return input_data
    except Exception as e:
        logging.error(f"Feature alignment error: {str(e)}")
        st.error("Feature configuration error. Please check input data.")
        st.stop()

def load_lottie(url):
    """Load Lottie animation with error handling"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.warning(f"Animation load error: {str(e)}")
        return None

# =================== ANIMATIONS ===================
animations = {
    'loading': load_lottie("https://assets8.lottiefiles.com/private_files/lf30_5ttqPi.json"),
    'churn': load_lottie("https://assets8.lottiefiles.com/packages/lf20_eeuhmcne.json"),
    'success': load_lottie("https://assets8.lottiefiles.com/packages/lf20_oftaxdnf.json")
}

# =================== CUSTOM STYLING ===================
st.markdown("""
    <style>
    .main {padding: 20px;}
    .sidebar .sidebar-content {background-color: #f0f2f6;}
    .stNumberInput, .stSelectbox {margin-bottom: 15px;}
    .metric-box {padding: 15px; border-radius: 10px; background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .highlight {background-color: #e6f7ff !important;}
    </style>
    """, unsafe_allow_html=True)

# =================== HEADER SECTION ===================
st.markdown("""
    <div style="background-color:#006d77;padding:25px;border-radius:10px;margin-bottom:25px;">
        <h1 style="color:white;text-align:center;margin:0;">📈 Telecom Customer Churn Predictor</h1>
        <p style="color:white;text-align:center;margin:5px 0 0;">Predict customer retention risks using advanced machine learning</p>
    </div>
    """, unsafe_allow_html=True)

# =================== SIDEBAR INPUTS ===================
with st.sidebar:
    st.header("📥 Customer Profile")
    with st.expander("ℹ️ Input Guide", expanded=True):
        st.markdown("""
            - **Age Group**: 1=18-25, 2=26-35, 3=36-45, 4=46-55, 5=55+
            - **Tariff Plan**: 1=Basic, 2=Premium
            - **Status**: 1=Active, 2=Inactive
            """)
    
    # Demographics Section
    with st.container():
        st.subheader("Demographics")
        age = st.number_input("Age", 15, 100, 30, help="Customer's current age")
        age_group = st.selectbox("Age Group", [1, 2, 3, 4, 5], format_func=lambda x: f"Group {x}")

    # Usage Metrics
    with st.container():
        st.subheader("Usage Metrics")
        subscription_length = st.number_input("Subscription Length (months)", 3, 60, 24,
                                            help="Duration of current subscription")
        charge_amount = st.slider("Charge Amount ($)", 0.0, 10.0, 5.0, 0.1,
                                help="Average monthly charge amount")
        seconds_of_use = st.number_input("Monthly Usage Seconds", 0, 50000, 5000,
                                       help="Total call duration per month")
        
    # Behavior Metrics
    with st.container():
        st.subheader("Behavior Metrics")
        frequency_of_use = st.number_input("Call Frequency", 0, 500, 50,
                                         help="Number of calls per month")
        frequency_of_sms = st.number_input("SMS Frequency", 0, 1000, 100,
                                         help="Number of SMS sent per month")
        distinct_called_numbers = st.number_input("Unique Contacts", 0, 500, 20,
                                                help="Distinct phone numbers contacted")

    tariff_plan = st.selectbox("Tariff Plan", [1, 2], format_func=lambda x: "Basic" if x == 1 else "Premium")
    status = st.selectbox("Account Status", [1, 2], format_func=lambda x: "Active" if x == 1 else "Inactive")

# =================== INPUT PROCESSING ===================
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
}).pipe(align_features)

# =================== MAIN CONTENT ===================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎯 Prediction Interface")
    
    if animations['loading']:
        st_lottie(animations['loading'], height=200, key="loading")
    
    if st.button("🚀 Analyze Churn Risk", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing customer profile..."):
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1]
                
                st.markdown(f"""
                    <div class="metric-box">
                        <h3 style="margin:0;color:#2a9d8f;">Prediction Confidence: {probability*100:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                if prediction[0] == 1:
                    st.markdown("""
                        <div style="background-color:#f94144;padding:20px;border-radius:10px;margin:20px 0;text-align:center;">
                            <h2 style="color:white;margin:0;">⚠️ High Churn Risk</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    if animations['churn']:
                        st_lottie(animations['churn'], height=200, key="churn")
                    with st.expander("📝 Recommended Actions"):
                        st.markdown("""
                            - Offer loyalty discount
                            - Provide personalized service checkup
                            - Schedule retention call
                            - Propose upgrade offer
                            """)
                else:
                    st.markdown("""
                        <div style="background-color:#2a9d8f;padding:20px;border-radius:10px;margin:20px 0;text-align:center;">
                            <h2 style="color:white;margin:0;">✅ Low Churn Risk</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    if animations['success']:
                        st_lottie(animations['success'], height=200, key="success")
                    with st.expander("📈 Retention Opportunities"):
                        st.markdown("""
                            - Suggest premium add-ons
                            - Offer referral bonus
                            - Provide usage insights
                            - Prolong subscription offer
                            """)
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            st.error("Analysis failed. Please check input values and try again.")

with col2:
    st.markdown("### 📚 Model Insights")
    with st.container():
        st.markdown("""
            <div class="metric-box">
                <h4 style="margin:0;">Model Performance</h4>
                <p style="margin:5px 0;">🎯 Accuracy: 94%</p>
                <p style="margin:5px 0;">📈 AUC-ROC: 0.88</p>
                <p style="margin:5px 0;">⚖️ F1 Score: 0.89</p>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander("🔍 Feature Importance"):
        st.markdown("""
            Top Predictive Factors:
            1. Charge Amount
            2. Frequency of Use
            3. Subscription Length
            4. Account Status
            5. Seconds of Use
            """)
    
    with st.expander("⚙️ Data Profile"):
        st.write("Current Input Values:")
        st.dataframe(input_data[model_features].style.applymap(lambda x: "background-color: #e6f7ff"), use_container_width=True)

# =================== FOOTER ===================
st.markdown("""
    <div style="background-color:#006d77;padding:15px;border-radius:10px;margin-top:30px;">
        <p style="color:white;text-align:center;margin:0;">
            Developed by Daniel Wanjala | 📧 dmwanjala254@gmail.com| Version 2.1
        </p>
    </div>
    """, unsafe_allow_html=True)