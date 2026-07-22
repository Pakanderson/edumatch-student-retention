import pandas as pd
import streamlit as st
from utils import apply_custom_styles, load_all_assets, init_session_state, clear_inputs

st.set_page_config(
    page_title="EduMatch Academic Advisory Suite", page_icon="🎓", layout="wide"
)

apply_custom_styles()
init_session_state()

model, scaler, kmeans, scaler_clustering, chunks, vectorizer, tfidf_matrix = (
    load_all_assets()
)

st.title("🎓 EduMatch: Predictive Student Retention Engine")
st.subheader("📋 Page 1: Advisor Profile Registration Input")

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
    ects_s1 = st.number_input("ECTS Credits Earned (Sem 1)", 0, 30, 12)
    grade_s1 = st.slider(
        "Grade Average (Sem 1) [1.0 Best to 5.0 Fail]", 1.0, 5.0, 3.8, 0.1
    )
    ects_s2 = st.number_input("ECTS Credits Earned (Sem 2)", 0, 30, 10)
    grade_s2 = st.slider(
        "Grade Average (Sem 2) [1.0 Best to 5.0 Fail]", 1.0, 5.0, 3.9, 0.1
    )

    submit_btn = st.form_submit_button("🚀 Run Prediction & RAG Analysis")

if st.button("🧹 Clear All Inputs"):
    clear_inputs()
    st.rerun()

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

    if (
        model is not None
        and scaler is not None
        and kmeans is not None
        and scaler_clustering is not None
    ):
        try:
            input_df_rf = input_df[scaler.feature_names_in_]
            scaled_rf = scaler.transform(input_df_rf)
            st.session_state.risk_pct = model.predict_proba(scaled_rf)[0][1] * 100

            input_df_km = input_df[scaler_clustering.feature_names_in_]
            scaled_km = scaler_clustering.transform(input_df_km)
            st.session_state.cluster_id = kmeans.predict(scaled_km)[0]

        except KeyError as e:
            st.error(
                f"❌ **Schema Mismatch Error:** Notebook features matrix out of alignment: {e}"
            )
            st.stop()
        except Exception as e:
            st.warning(f"⚠️ Computational calculation trace failure: {e}")
            st.session_state.risk_pct = None
            st.session_state.cluster_id = 1 if (ects_s1 + ects_s2) < 30 else 0
    else:
        st.session_state.risk_pct = None
        st.session_state.cluster_id = 1 if (ects_s1 + ects_s2) < 30 else 0

    st.success(
        "✅ Student Profile Loaded! Switch to **02 Live Analytics** in the sidebar to review results."
    )
