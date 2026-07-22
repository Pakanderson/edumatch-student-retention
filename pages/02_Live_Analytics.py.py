import streamlit as st
from utils import apply_custom_styles, init_session_state

st.set_page_config(page_title="Live Analytics Engine", page_icon="⚡", layout="wide")

apply_custom_styles()
init_session_state()

st.title("⚡ Live Analytics Engine")

if st.session_state.cached_student is not None:
    c = st.session_state.cached_student
    cluster_id = st.session_state.cluster_id

    cluster_labels = {
        0: "Cluster 0: High Academic Progress with Structural Risk Factors",
        1: "Cluster 1: Moderate Credit Accumulation and Study-Load Risk",
        2: "Cluster 2: Early Non-Engagement Profile: Younger Male Students",
        3: "Cluster 3: Employed Student Study-Work Pressure Profile",
        4: "Cluster 4: International Student Transition and Credit-Progress Risk",
        5: "Cluster 5: BAföG Recipient Financial-Support and Progression Risk",
        6: "Cluster 6: Mature Student High-Performance with Retention Risk",
        7: "Cluster 7: Stable Academic Progress with Socio-Economic Vulnerability",
        8: "Cluster 8: Early Non-Engagement Profile: Mature Students",
        9: "Cluster 9: BAföG Recipient Declining Academic-Progress Profile",
        10: "Cluster 10: Employed Mature Student Study-Work Pressure Profile",
        11: "Cluster 11: Mid-Programme Semester-Two Academic Decline Profile",
        12: "Cluster 12: Working Master’s Student Academic-Progress Decline Profile",
        13: "Cluster 13: High ECTS Accumulation with General Retention Risk",
        14: "Cluster 14: Early Non-Engagement Profile: Younger Female Students",
    }

    # Dynamic Heuristic Calibrations
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

    if st.session_state.risk_pct is not None:
        final_risk_pct = st.session_state.risk_pct
    else:
        final_risk_pct = (academic_score * 0.70) + (socioeconomic_score * 0.30)
        final_risk_pct = min(98.5, max(4.5, final_risk_pct))

    # Domain Hard Override
    if c["grade_s1"] <= 2.0 and c["grade_s2"] <= 2.0:
        final_risk_pct = min(final_risk_pct, 25.0)

    if final_risk_pct >= 45.0:
        st.error(
            f"### ⚠️ HIGH RETENTION ALERT: **{final_risk_pct:.1f}% Attrition Probability** (Threshold: 45.0%)"
        )
    else:
        st.success(
            f"### ✅ **Stable Standing Profile: {final_risk_pct:.1f}% Attrition Probability**"
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
            "🔴 Impacted by integration or funding markers."
            if socioeconomic_score >= 50.0
            else "🟢 Structural context clear."
        )

    st.markdown("---")
    st.markdown(
        f"**👥 Cohort Profile Focus:** {cluster_labels.get(cluster_id, 'Specialized Framework Segment Overview')}"
    )

else:
    st.info(
        "ℹ️ Fill out student parameters on Page 1 (**main_app.py**) and click **Run Prediction & RAG Analysis**."
    )
