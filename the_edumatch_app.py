import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ===========================================================================
# STRUCTURAL INITIALIZATION & SYSTEM GATEWAYS
# ===========================================================================
load_dotenv(override=True)

raw_key = os.environ.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
GROQ_API_KEY = None
if raw_key:
    GROQ_API_KEY = (
        raw_key.strip()
        .replace('"', "")
        .replace("'", "")
        .replace("\r", "")
        .replace("\n", "")
    )

st.set_page_config(
    page_title="EduMatch Academic Advisory Suite", page_icon="🎓", layout="wide"
)


# ===========================================================================
# ASSET PACK LOADING & RAG ENGINE CACHING
# ===========================================================================
@st.cache_resource(show_spinner=False)
def load_all_assets():
    model, scaler, kmeans, scaler_clustering = None, None, None, None
    try:
        model = joblib.load("models/german_retention_model.pkl")
        scaler = joblib.load("models/clustering scaler.pkl")
        kmeans = joblib.load("models/kmeans_model.pkl")
        scaler_clustering = joblib.load("models/clustering_scaler.pkl")
    except Exception as e:
        st.error(f"❌ Error loading production system files: {e}")

    chunks = []
    if os.path.exists("raw_extracted_po.txt"):
        with open("raw_extracted_po.txt", "r", encoding="utf-8") as f:
            chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]
    elif os.path.exists("examination_regulations.txt"):
        with open("examination_regulations.txt", "r", encoding="utf-8") as f:
            chunks = [
                c.strip() for c in f.read().split("=== CLAUSE START ===") if c.strip()
            ]

    if not chunks:
        chunks = [
            "[PO-101] General Examination Regulations: Fees paid prior to semester deadlines.",
            "[PO-201] Grading Matrix: Grades over 3.5 flag monitoring.",
            "[PO-301] Minimum Credits: 15 ECTS per semester required.",
            "[PO-302] Repetition: Three attempts allowed for core modules.",
        ]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunks)

    return model, scaler, kmeans, scaler_clustering, chunks, vectorizer, tfidf_matrix


model, scaler, kmeans, scaler_clustering, chunks, vectorizer, tfidf_matrix = (
    load_all_assets()
)

# ===========================================================================
# SESSION STATE LIFECYCLE MANAGEMENT
# ===========================================================================
if "form_key" not in st.session_state:
    st.session_state.form_key = 0
if "cached_student" not in st.session_state:
    st.session_state.cached_student = None
if "risk_pct" not in st.session_state:
    st.session_state.risk_pct = None
if "cluster_id" not in st.session_state:
    st.session_state.cluster_id = None
if "sandbox_response" not in st.session_state:
    st.session_state.sandbox_response = None
if "sandbox_chunks" not in st.session_state:
    st.session_state.sandbox_chunks = None


def clear_inputs():
    st.session_state.form_key += 1
    st.session_state.cached_student = None
    st.session_state.risk_pct = None
    st.session_state.cluster_id = None
    st.session_state.sandbox_response = None
    st.session_state.sandbox_chunks = None


st.title("🎓 EduMatch: Predictive Student Retention and Prescriptive Analytics")
col1, col2 = st.columns([1, 1.2])

# ===========================================================================
# ADVISOR REGISTRATION INPUT PANEL (COL1)
# ===========================================================================
with col1:
    st.header("📋 Advisor Input Panel")
    with st.form(key=f"input_form_{st.session_state.form_key}"):

        st.subheader("🌍 Socio-economic Indicators")
        gender = st.selectbox("Gender", ["Female", "Male"])
        residency = st.selectbox(
            "Residency Classification",
            ["EU / Domestic Student", "Non-EU International Track"],
        )
        bafoeg = st.selectbox("BAföG Recipient Status", ["No", "Yes (Recipient)"])
        student_job = st.selectbox(
            "Employment Configuration",
            [
                "No Job (Full-Time Study Focus)",
                "Balancing Student Job / Part-Time Work",
            ],
        )
        accommodation = st.selectbox(
            "Accommodation Stability",
            ["Stable Housing Structure", "Unstable Accommodation Arrangement"],
        )

        st.subheader("📚 Academic Milestones")
        is_master = st.selectbox(
            "Enrolled Degree Level",
            ["Bachelor Level Degree Program", "Master Level Degree program"],
        )
        ects_s1 = st.number_input("ECTS Credits Earned (Sem 1)", 0, 40, 12)
        grade_s1 = st.slider(
            "Grade Average (Sem 1) [1.0 Best to 5.0 Fail]", 1.0, 5.0, 3.8, 0.1
        )
        ects_s2 = st.number_input("ECTS Credits Earned (Sem 2)", 0, 40, 10)
        grade_s2 = st.slider(
            "Grade Average (Sem 2) [1.0 Best to 5.0 Fail]", 1.0, 5.0, 3.9, 0.1
        )

        submit_btn = st.form_submit_button("🚀 Run Prediction & RAG Analysis")

    if st.button("🧹 Clear All Inputs"):
        clear_inputs()
        st.rerun()

    # --- REACTIVE MATRIX CALCULATOR TRIGGER ---
    if submit_btn:
        st.session_state.risk_pct = None
        st.session_state.cached_student = {
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

        input_dict = {
            "BAfoeg_Status": 1 if bafoeg == "Yes (Recipient)" else 0,
            "Residency_Status": 1 if "Non-EU" in residency else 0,
            "Student_Job": 1 if "Job" in student_job else 0,
            "Accommodation_Status": 1 if "Unstable" in accommodation else 0,
            "Gender": 1 if gender == "Male" else 0,
            "Is_Master": 1 if "Master" in is_master else 0,
            "ECTS_Earned_Sem1": ects_s1,
            "Grade_Avg_Sem1": grade_s1,
            "ECTS_Earned_Sem2": ects_s2,
            "Grade_Avg_Sem2": grade_s2,
        }
        input_df = pd.DataFrame([input_dict])

        if all([model, scaler, kmeans, scaler_clustering]):
            try:
                scaled_rf = scaler.transform(input_df[scaler.feature_names_in_])
                st.session_state.risk_pct = model.predict_proba(scaled_rf)[0][1] * 100
                scaled_km = scaler_clustering.transform(
                    input_df[scaler_clustering.feature_names_in_]
                )
                st.session_state.cluster_id = kmeans.predict(scaled_km)[0]
            except Exception:
                st.session_state.risk_pct = None
        st.rerun()

# ===========================================================================
# LIVE RETENTION INTEGRITY ANALYTICS CORE (COL2)
# ===========================================================================
with col2:
    st.header("⚡ Live Analytics Engine")
    if st.session_state.cached_student:
        c = st.session_state.cached_student

        # Calculate Heuristic Scores
        academic_score = min(
            100.0,
            max(
                5.0,
                (max(0, 30 - c["ects_s1"]) * 1.5 + max(0, 30 - c["ects_s2"]) * 1.5)
                + (
                    max(0.0, c["grade_s1"] - 1.0) * 10.0
                    + max(0.0, c["grade_s2"] - 1.0) * 10.0
                ),
            ),
        )
        socioeconomic_score = min(
            100.0,
            10.0
            + (25.0 if "Non-EU" in c["residency"] else 0)
            + (20.0 if "Job" in c["student_job"] else 0),
        )

        final_risk_pct = (
            st.session_state.risk_pct
            if st.session_state.risk_pct is not None
            else (academic_score * 0.7 + socioeconomic_score * 0.3)
        )

        # Domain Expert Override: For high academic performance
        if c["grade_s1"] <= 1.5 and c["grade_s2"] <= 1.5:
            final_risk_pct = min(final_risk_pct, 25.0)

        if final_risk_pct >= 40.0:
            st.error(
                f"### ⚠️ HIGH RETENTION ALERT: **{final_risk_pct:.1f}% Attrition Probability**"
            )
        else:
            st.success(
                f"### ✅ **Stable Standing Profile: {final_risk_pct:.1f}% Attrition Probability**"
            )
