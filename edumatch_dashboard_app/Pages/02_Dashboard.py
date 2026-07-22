import streamlit as st
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from utils import apply_custom_styles
import streamlit as st
from utils import apply_custom_styles, load_all_assets

# 1. Apply styles immediately
apply_custom_styles()

# 2. Load assets (this will only run once and then use the cache)
model, scaler, kmeans = load_all_assets()

# Call this immediately after st.set_page_config
apply_custom_styles()

st.title("🎓 EduMatch Dashboard")
# The rest of your page code...
st.set_page_config(page_title="Advisor Dashboard", layout="wide")

# Load your production assets here using st.cache_resource
@st.cache_resource
def load_assets():
    model = joblib.load("models/german_retention_model.pkl")
    scaler = joblib.load("models/clustering scaler.pkl")
    kmeans = joblib.load("models/kmeans_model.pkl")
    return model, scaler, kmeans

model, scaler, kmeans = load_assets()

st.title("📊 Advisor Dashboard")

if st.session_state.student_data:
    c = st.session_state.student_data
    # Reconstruct input dataframe
    input_dict = {"BAfoeg_Status": 1 if c['bafoeg'] == "Yes (Recipient)" else 0, ...} # Map all 11 features
    input_df = pd.DataFrame([input_dict])
    
    # Run predictions
    risk_pct = model.predict_proba(scaler.transform(input_df))[0][1] * 100
    st.metric("Attrition Risk", f"{risk_pct:.1f}%")
    # ... Add your plotting code here ...
else:
    st.warning("Please submit data on the main page first.")