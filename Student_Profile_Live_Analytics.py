import streamlit as st
import pandas as pd
from utils import (
    apply_custom_styles,
    init_session_state,
    clear_inputs,
    load_all_assets,
    CLUSTER_LABELS,
)

# 1. Page Configuration
st.set_page_config(
    page_title="Student Profile & LIVE ANALYTICS", page_icon="🎓", layout="wide"
)

# 2. System Initialization
apply_custom_styles()
init_session_state()

model, scaler, kmeans, scaler_clustering, chunks, vectorizer, tfidf_matrix = (
    load_all_assets()
)

st.title("🎓 STUDENT PROFILE AND LIVE ANALYTICS")
st.markdown("---")

col1, col2 = st.columns([1, 1.2])

# Helper lists for index lookups
GENDER_OPTS = ["Female", "Male"]
RESIDENCY_OPTS = ["EU / Domestic Student", "Non-EU International Track"]
BAFOEG_OPTS = ["No", "Yes (Recipient)"]
JOB_OPTS = ["No Job (Full-Time Study Focus)", "Balancing Student Job / Part-Time Work"]
ACCOM_OPTS = ["Stable Housing Structure", "Unstable Accommodation Arrangement"]
DEGREE_OPTS = ["Bachelor Level Degree Program", "Master Level Degree program"]

# Ensure default cached dictionary structure exists for form persistence
if st.session_state.cached_student is None:
    current = {
        "gender": "Female",
        "residency": "EU / Domestic Student",
        "bafoeg": "No",
        "student_job": "No Job (Full-Time Study Focus)",
        "accommodation": "Stable Housing Structure",
        "is_master": "Bachelor Level Degree Program",
        "ects_s1": 12,
        "grade_s1": 3.8,
        "ects_s2": 10,
        "grade_s2": 3.9,
    }
else:
    current = st.session_state.cached_student

# ===========================================================================
# ADVISOR REGISTRATION INPUT PANEL (COL1)
# ===========================================================================
with col1:
    st.header("📋 Student Profile")
    with st.form(key=f"input_form_{st.session_state.form_key}"):

        st.subheader("🌍 Socio-economic Indicators")

        gender_idx = (
            GENDER_OPTS.index(current["gender"])
            if current["gender"] in GENDER_OPTS
            else 0
        )
        gender = st.selectbox("Gender", GENDER_OPTS, index=gender_idx)

        res_idx = (
            RESIDENCY_OPTS.index(current["residency"])
            if current["residency"] in RESIDENCY_OPTS
            else 0
        )
        residency = st.selectbox(
            "Residency Classification", RESIDENCY_OPTS, index=res_idx
        )

        baf_idx = (
            BAFOEG_OPTS.index(current["bafoeg"])
            if current["bafoeg"] in BAFOEG_OPTS
            else 0
        )
        bafoeg = st.selectbox("BAföG Recipient Status", BAFOEG_OPTS, index=baf_idx)

        job_idx = (
            JOB_OPTS.index(current["student_job"])
            if current["student_job"] in JOB_OPTS
            else 0
        )
        student_job = st.selectbox("Employment Configuration", JOB_OPTS, index=job_idx)

        acc_idx = (
            ACCOM_OPTS.index(current["accommodation"])
            if current["accommodation"] in ACCOM_OPTS
            else 0
        )
        accommodation = st.selectbox(
            "Accommodation Stability", ACCOM_OPTS, index=acc_idx
        )

        st.subheader("📚 Academic Milestones")
        deg_idx = (
            DEGREE_OPTS.index(current["is_master"])
            if current["is_master"] in DEGREE_OPTS
            else 0
        )
        is_master = st.selectbox("Enrolled Degree Level", DEGREE_OPTS, index=deg_idx)

        ects_s1 = st.number_input(
            "ECTS Credits Earned (Sem 1)", 0, 30, value=int(current["ects_s1"])
        )
        grade_s1 = st.slider(
            "Grade Average (Sem 1) [1.0 Best to 5.0 Fail]",
            1.0,
            5.0,
            value=float(current["grade_s1"]),
            step=0.1,
        )

        ects_s2 = st.number_input(
            "ECTS Credits Earned (Sem 2)", 0, 30, value=int(current["ects_s2"])
        )
        grade_s2 = st.slider(
            "Grade Average (Sem 2) [1.0 Best to 5.0 Fail]",
            1.0,
            5.0,
            value=float(current["grade_s2"]),
            step=0.1,
        )

# --- PREDICTION LOGIC ---
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

            except Exception:
                st.session_state.risk_pct = None
                st.session_state.cluster_id = 1 if (ects_s1 + ects_s2) < 30 else 0
        else:
            # Fallback if models are missing
            st.session_state.risk_pct = None
            st.session_state.cluster_id = 1 if (ects_s1 + ects_s2) < 30 else 0

        # Rerun to update the UI with new state
        st.rerun()
        # Store input selections into session state so returning to this page retains all choices
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

            except Exception:
                st.session_state.risk_pct = None
                st.session_state.cluster_id = 1 if (ects_s1 + ects_s2) < 30 else 0
        else:
            # Fallback when models/scalers aren't loaded
            st.session_state.risk_pct = None
            st.session_state.cluster_id = 1 if (ects_s1 + ects_s2) < 30 else 0

        st.rerun()

# ===========================================================================
# LIVE RETENTION INTEGRITY ANALYTICS CORE (COL2)
# ===========================================================================
with col2:
    st.header("⚡ Live Analytics Engine")
    if st.session_state.cached_student is not None:
        c = st.session_state.cached_student
        cluster_id = st.session_state.cluster_id

        # 1. Compute Operational Sub-Driver Scores
        s1_deficit = max(0, 30 - c["ects_s1"])
        s2_deficit = max(0, 30 - c["ects_s2"])
        ects_penalty_score = (s1_deficit * 1.5) + (s2_deficit * 1.5)

        g1_stress = max(0.0, c["grade_s1"] - 1.0) * 10.0
        g2_stress = max(0.0, c["grade_s2"] - 1.0) * 10.0
        grade_penalty_score = g1_stress + g2_stress

        academic_score = min(100.0, max(5.0, ects_penalty_score + grade_penalty_score))

        socio_base = 10.0
        if "Non-EU" in c["residency"]:
            socio_base += 25.0
        if "Job" in c["student_job"]:
            socio_base += 20.0
        if "Unstable" in c["accommodation"]:
            socio_base += 20.0
        if c["bafoeg"] == "No":
            socio_base += 15.0
        socioeconomic_score = min(100.0, socio_base)

        # 2. Transparent Weighted Attrition Calculation (60% Academic / 40% Socioeconomic)
        heuristic_risk = (academic_score * 0.60) + (socioeconomic_score * 0.40)
        final_risk_pct = min(98.5, max(4.5, heuristic_risk))

        # 3. Dynamic Visual Alert Banner (45.0% Threshold)
        if final_risk_pct >= 45.0:
            st.error(
                f"### ⚠️ HIGH RETENTION ALERT: **{final_risk_pct:.1f}% Attrition Probability** (Threshold: 45.0%)"
            )
        else:
            st.success(
                f"### ✅ **Stable Standing Profile: {final_risk_pct:.1f}% Attrition Probability** (Threshold: 45.0%)"
            )

        st.markdown("#### 📊 Risk Driver Deconstruction")
        sub_col1, sub_col2 = st.columns(2)

        with sub_col1:
            st.metric(
                label="📚 Operational Academic Risk Score",
                value=f"{academic_score:.1f}%",
            )
            st.caption(
                "🔴 Driven by compounding credit deficits."
                if academic_score >= 50.0
                else "🟢 Progress metrics stable."
            )

        with sub_col2:
            st.metric(
                label="🌍 Socioeconomic & Environmental Strain",
                value=f"{socioeconomic_score:.1f}%",
            )
            st.caption(
                "🔴 Impacted by work limits or funding."
                if socioeconomic_score >= 50.0
                else "🟢 Structural context clear."
            )

        st.markdown("---")
        st.markdown(
            f"**👥 Cohort Profile Focus:** {CLUSTER_LABELS.get(cluster_id, 'Specialized Framework Segment Overview')}"
        )

        st.info(
            "💡 **Next Step:** Select **02 Policy Advisory** in the sidebar menu to view RAG regulatory reports."
        )
    else:
        st.info(
            "ℹ️ Fill out student parameters on the left panel and click **Run Prediction & Live Analytics**."
        )
