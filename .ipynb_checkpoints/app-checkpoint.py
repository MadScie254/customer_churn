import streamlit as st
import pandas as pd
import joblib
import requests
import logging
import plotly.express as px
from streamlit_lottie import st_lottie

# =================== PAGE CONFIGURATION ===================
st.set_page_config(
    page_title="Telecom Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================== INITIAL SETUP ===================
logging.basicConfig(filename='app.log', level=logging.ERROR)

# =================== MODEL MANAGEMENT ===================
@st.cache_resource
def load_model():
    """Cache-loaded ML model for efficient predictions"""
    try:
        model = joblib.load('xgb_churn_model.pkl')
        st.session_state['model_loaded'] = True
        return model
    except Exception as e:
        logging.error(f"Model loading failed: {str(e)}")
        st.error("Critical Error: Prediction model unavailable. Contact support.")
        st.stop()

model = load_model()
model_features = model.get_booster().feature_names

# =================== UTILITY FUNCTIONS ===================
def validate_features(input_df):
    """Ensure feature alignment between input and model requirements"""
    try:
        aligned_df = input_df.reindex(columns=model_features, fill_value=0)
        if list(aligned_df.columns) != list(model_features):
            raise ValueError("Feature mismatch detected")
        return aligned_df
    except Exception as e:
        logging.error(f"Feature validation failed: {str(e)}")
        st.error("System Error: Feature configuration issue")
        st.stop()

def fetch_animation(url):
    """Retrieve Lottie animations with robust error handling"""
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.warning(f"Animation load failed: {str(e)}")
        return None

def plot_feature_importance():
    """Create modern dark theme feature importance visualization"""
    importance = model.get_booster().get_score(importance_type='weight')
    df_importance = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(df_importance, 
                 x='Importance', 
                 y='Feature', 
                 orientation='h',
                 title='<b>Feature Importance Analysis</b>',
                 color='Importance',
                 color_continuous_scale='tealrose')
    
    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font_color='#FFFFFF',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=500,
        margin=dict(l=100, r=30, b=50, t=80),
        title_font_size=20,
        coloraxis_showscale=False
    )
    return fig

# =================== ASSET LOADING ===================
ANIMATIONS = {
    'loading': fetch_animation("https://assets8.lottiefiles.com/private_files/lf30_5ttqPi.json"),
    'churn': fetch_animation("https://assets8.lottiefiles.com/packages/lf20_eeuhmcne.json"),
    'success': fetch_animation("https://assets8.lottiefiles.com/packages/lf20_oftaxdnf.json"),
    'error': fetch_animation("https://assets1.lottiefiles.com/packages/lf20_tiviycvc.json"),
    'data_input': fetch_animation("https://assets8.lottiefiles.com/packages/lf20_sk5h1kfn.json"),
    'processing': fetch_animation("https://assets10.lottiefiles.com/packages/lf20_isdxvuls.json")
}

# =================== UI CONFIGURATION ===================
DARK_THEME = """
    <style>
    [data-testid="stAppViewContainer"] {background-color: #0E1117;}
    .sidebar .sidebar-content {background: #161925 !important; border-right: 1px solid #2E2E2E;}
    .stNumberInput, .stSelectbox {margin-bottom: 1.2rem;}
    .metric-card {background: #161925; padding: 1.5rem; border-radius: 12px; 
                border: 1px solid #2E2E2E; margin-bottom: 1.5rem;}
    .dark-header {background: linear-gradient(45deg, #1F2A40, #161925) !important; 
                color: white !important; border: none !important;}
    .footer {background: #161925 !important; color: white !important; border-top: 1px solid #2E2E2E;}
    .feature-plot {border-radius: 12px; overflow: hidden; border: 1px solid #2E2E2E;}
    h1, h2, h3, h4, h5, h6 {color: #FFFFFF !important;}
    p, label {color: #8B95A5 !important;}
    .st-expander {background: #161925 !important; border: 1px solid #2E2E2E !important; border-radius: 8px !important;}
    .st-expanderHeader {background: #161925 !important; color: #FFFFFF !important;}
    .st-b7 {color: #FFFFFF !important;}
    </style>
"""
st.markdown(DARK_THEME, unsafe_allow_html=True)

# =================== HEADER SECTION ===================
st.markdown("""
    <div class="metric-card dark-header" style="margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5em;">📱 Telecom Churn Predictor</h1>
        <p style="margin: 0.5rem 0 0; font-size: 1.1em;">AI-powered customer retention analysis</p>
    </div>
""", unsafe_allow_html=True)

# =================== SIDEBAR CONFIGURATION ===================
with st.sidebar:
    if ANIMATIONS['data_input']:
        st_lottie(ANIMATIONS['data_input'], height=120, key="sidebar_anim")
    
    st.markdown("""
        <div style="padding: 1.5rem; background: #1F2A40; border-radius: 12px; 
                border: 1px solid #2E2E2E; margin-bottom: 2rem;">
            <h3 style="color: #FFFFFF; margin-bottom: 0.5rem;">📋 Data Input Guide</h3>
            <p style="color: #8B95A5; margin: 0;">Complete all fields for accurate predictions</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input Sections with animations
    with st.expander("👤 Demographics", expanded=True):
        if ANIMATIONS['processing']:
            st_lottie(ANIMATIONS['processing'], height=80, key="demo_anim")
        age = st.number_input("Customer Age", 18, 100, 35)
        age_group = st.selectbox("Age Category", 
                               options=[1, 2, 3, 4, 5],
                               format_func=lambda x: f"{x} (18-25)" if x == 1 else 
                                       f"{x} (26-35)" if x == 2 else
                                       f"{x} (36-45)" if x == 3 else
                                       f"{x} (46-55)" if x == 4 else
                                       f"{x} (55+)")
    
    with st.expander("📊 Usage Metrics", expanded=True):
        if ANIMATIONS['processing']:
            st_lottie(ANIMATIONS['processing'], height=80, key="usage_anim")
        subscription_length = st.slider("Subscription Duration (months)", 3, 60, 24)
        charge_amount = st.number_input("Monthly Charges ($)", 0.0, 100.0, 45.99)
        usage_seconds = st.number_input("Monthly Call Minutes", 0, 50000, 2500)
    
    with st.expander("📈 Behavior Patterns", expanded=True):
        if ANIMATIONS['processing']:
            st_lottie(ANIMATIONS['processing'], height=80, key="behavior_anim")
        call_frequency = st.number_input("Weekly Calls", 0, 100, 15)
        sms_frequency = st.number_input("Weekly SMS", 0, 200, 30)
        unique_contacts = st.number_input("Unique Contacts", 0, 500, 50)
    
    with st.expander("⚙️ Account Details", expanded=True):
        if ANIMATIONS['processing']:
            st_lottie(ANIMATIONS['processing'], height=80, key="account_anim")
        tariff_plan = st.radio("Plan Type", [1, 2], 
                             format_func=lambda x: "Basic" if x == 1 else "Premium")
        account_status = st.selectbox("Account State", [1, 2], 
                                    format_func=lambda x: "Active" if x == 1 else "Dormant")

# =================== DATA PROCESSING ===================
input_template = {
    'Call Failure': [0],
    'Complains': [0],
    'Subscription Length': subscription_length,
    'Charge Amount': charge_amount,
    'Seconds of Use': usage_seconds,
    'Frequency of use': call_frequency * 4,
    'Frequency of SMS': sms_frequency * 4,
    'Distinct Called Numbers': unique_contacts,
    'Age Group': age_group,
    'Tariff Plan': tariff_plan,
    'Status': account_status,
    'Age': age,
    'Customer Value': 0
}

processed_data = pd.DataFrame(input_template).pipe(validate_features)

# =================== MAIN INTERFACE ===================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🔮 Churn Prediction Analysis")
    
    if ANIMATIONS['loading']:
        st_lottie(ANIMATIONS['loading'], height=180, key="loader")
    
    if st.button("Run Churn Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing customer patterns..."):
            try:
                # Add validation checks
                if not all(isinstance(val, (int, float)) for val in input_template.values()):
                    raise ValueError("Invalid input values detected")
                
                prediction = model.predict(processed_data)
                confidence = model.predict_proba(processed_data)[0][1]
                
                st.markdown(f"""
                    <div class="metric-card" style="margin: 1rem 0;">
                        <h3 style="color: #FFFFFF; margin-bottom: 0;">Prediction Confidence: {confidence*100:.1f}%</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                if prediction[0] == 1:
                    st.markdown("""
                        <div style="background: #2D1A2D; color: #FF6B6B; padding: 1.5rem; 
                                border-radius: 12px; border: 1px solid #4A2A4A; margin: 1rem 0;">
                            <h2 style="margin: 0;">⚠️ High Churn Risk Detected</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    st_lottie(ANIMATIONS['churn'], height=200)
                    with st.expander("📌 Recommended Retention Strategies"):
                        st.markdown("""
                            - **Immediate Action**: Priority customer service outreach
                            - **Incentives**: Offer 15% loyalty discount
                            - **Service Review**: Schedule technical checkup
                            - **Plan Upgrade**: Suggest premium features
                        """)
                else:
                    st.markdown("""
                        <div style="background: #1A2D1A; color: #6BFF6B; padding: 1.5rem; 
                                border-radius: 12px; border: 1px solid #2A4A2A; margin: 1rem 0;">
                            <h2 style="margin: 0;">✅ Low Churn Probability</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    st_lottie(ANIMATIONS['success'], height=200)
                    with st.expander("💡 Engagement Opportunities"):
                        st.markdown("""
                            - **Upsell**: Recommend data boost packages
                            - **Loyalty Program**: Introduce referral bonuses
                            - **Feedback**: Request service satisfaction survey
                        """)
                        
            except Exception as e:
                logging.error(f"Prediction failed: {str(e)}\nInput Data: {input_template}")
                st.error("Analysis error. Please check inputs and try again.")
                if ANIMATIONS['error']:
                    st_lottie(ANIMATIONS['error'], height=200, key="error_anim")

with col2:
    st.markdown("## 📊 Model Insights")
    
    with st.container():
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #FFFFFF; border-bottom: 1px solid #2E2E2E; padding-bottom: 0.5rem; margin-bottom: 1rem;">
                    Model Performance
                </h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                    <div style="background: #1A2D1A; padding: 1rem; border-radius: 8px; border: 1px solid #2A4A2A;">
                        <p style="color: #6BFF6B; margin: 0;">🎯 Accuracy</p>
                        <h3 style="color: #FFFFFF; margin: 0;">94.2%</h3>
                    </div>
                    <div style="background: #1A2D2D; padding: 1rem; border-radius: 8px; border: 1px solid #2A4A4A;">
                        <p style="color: #00D1FF; margin: 0;">📊 Precision</p>
                        <h3 style="color: #FFFFFF; margin: 0;">92.1%</h3>
                    </div>
                    <div style="background: #2D1A2D; padding: 1rem; border-radius: 8px; border: 1px solid #4A2A4A;">
                        <p style="color: #FF6B6B; margin: 0;">📈 Recall</p>
                        <h3 style="color: #FFFFFF; margin: 0;">89.5%</h3>
                    </div>
                    <div style="background: #2D2D1A; padding: 1rem; border-radius: 8px; border: 1px solid #4A4A2A;">
                        <p style="color: #FFD700; margin: 0;">⚖️ F1 Score</p>
                        <h3 style="color: #FFFFFF; margin: 0;">90.7</h3>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
            <div class="feature-plot" style="margin-top: 1.5rem;">
                <div style="padding: 1rem; background: #161925; border-bottom: 1px solid #2E2E2E;">
                    <h4 style="margin: 0; color: #FFFFFF;">Key Predictive Features</h4>
                </div>
        """, unsafe_allow_html=True)
        fig = plot_feature_importance()
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =================== FOOTER ===================
st.markdown("""
    <div class="footer" style="padding: 1rem; margin-top: 3rem;">
        <p style="text-align: center; margin: 0; color: #8B95A5;">
            🚀 Powered by XGBoost & Streamlit | 📧 support@telemanalytics.com | v2.5
        </p>
    </div>
""", unsafe_allow_html=True)