import streamlit as st
import os
from dotenv import load_dotenv
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

st.set_page_config(page_title="EduMatch Engine", layout="wide")

# Initialize session state
if "student_data" not in st.session_state:
    st.session_state.student_data = None

st.title("🎓 EduMatch: Predictive Engine")

with st.form("input_form"):
    st.header("Step 1: Student Profile Telemetry")
    gender = st.selectbox("Gender", ["Female", "Male"])
    residency = st.selectbox(
        "Residency Classification",
        ["EU / Domestic Student", "Non-EU International Track"],
    )
    bafoeg = st.selectbox("BAföG Recipient Status", ["No", "Yes (Recipient)"])
    student_job = st.selectbox(
        "Employment Configuration",
        ["No Job (Full-Time Study Focus)", "Balancing Student Job / Part-Time Work"],
    )
    accommodation = st.selectbox(
        "Accommodation Stability",
        ["Stable Housing Structure", "Unstable Accommodation Arrangement"],
    )
    is_master = st.selectbox(
        "Enrolled Degree Level",
        ["Bachelor Level Degree Program", "Master Level Degree program"],
    )

    ects_s1 = st.number_input("ECTS Credits Earned (Sem 1)", 0, 30, 12)
    grade_s1 = st.slider(
        "Grade Average (Sem 1) [1.0 Best to 5.0 Fail]", 1.0, 5.0, 3.8, 0.1
    )
    ects_s2 = st.number_input("ECTS Credits Earned (Sem 2)", 0, 30, 10)
    grade_s2 = st.slider(
        "Grade Average (Sem 2) [1.0 Best to 5.0 Fail]", 1.0, 5.0, 3.9, 0.1
    )

    submit = st.form_submit_button("Run Prediction")

if submit:
    st.session_state.student_data = {
        "bafoeg": bafoeg,
        "residency": residency,
        "student_job": student_job,
        "accommodation": accommodation,
        "gender": gender,
        "is_master": is_master,
        "ects_s1": ects_s1,
        "grade_s1": grade_s1,
        "ects_s2": ects_s2,
        "grade_s2": grade_s2,
    }
    st.success("Data stored. Go to the Dashboard page!")
