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

# Bulk-strip any trailing spaces, quotes, or shell artifacts from your API Key safely
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

# Enforce app-wide layout grid spacing constraints immediately prior to loading assets
st.set_page_config(
    page_title="EduMatch Academic Advisory Suite", page_icon="🎓", layout="wide"
)

#  PREMIUM DASHBOARD VISUAL ENGINE: CENTRALIZED CSS 
st.markdown(
    """
<style>
    /* 1. Center the master application title */
    h1 {
        text-align: center !important;
        font-size: 2.55rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* 2. Global presentation font scaling for compact viewing */
    html, body, [data-testid="stWidgetLabel"] p, .stSelectbox div, .stMarkdown p {
        font-size: 1.15rem !important;
    }
    
    /* 3. Subheader emphasis */
    h3, h4 {
        font-size: 1.4rem !important;
        font-weight: bold !important;
    }

    /* 4. Target and style the main prediction submit button inside the form container */
    .stFormSubmitButton > button {
        background-color: #2e7d32 !important; 
        color: white !important;              
        font-size: 1.25rem !important;
        font-weight: bold !important;
        height: 3em !important;
        width: 100% !important;
        border-radius: 8px !important;
        border: none !important;
    }
    .stFormSubmitButton > button:hover {
        background-color: #1b5e20 !important; 
        color: #ffffff !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ===========================================================================
# ASSET PACK LOADING & RAG ENGINE CACHING
# ===========================================================================
@st.cache_resource(show_spinner=False)
def load_all_assets():
    """
    Loads machine learning pipeline artifacts, scalers, and text chunks,
    automatically fitting text structures for out-of-sample RAG queries.
    """
    model, scaler, kmeans, scaler_clustering = None, None, None, None
    try:
        # Load the Supervised Random Forest Classifier Model
        model = joblib.load("models/german_retention_model.pkl")

        # Load the primary pipeline RobustScaler used for the main dataset split
        scaler = joblib.load("models/clustering scaler.pkl")

        # Load the Unsupervised K-Means Segmentation Model
        kmeans = joblib.load("models/kmeans_model.pkl")

        # Load the dedicated sub-scaler used for K-Means distance calculations
        scaler_clustering = joblib.load("models/clustering_scaler.pkl")

    except Exception as e:
        st.error(
            f"❌ Error loading production system files from models/ directory: {e}"
        )

    # --- REGULATORY TEXT INGESTION DISK SCANNER ---
    chunks = []
    if os.path.exists("raw_extracted_po.txt"):
        with open("raw_extracted_po.txt", "r", encoding="utf-8") as f:
            chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]
    elif os.path.exists("examination_regulations.txt"):
        with open("examination_regulations.txt", "r", encoding="utf-8") as f:
            chunks = [
                c.strip() for c in f.read().split("=== CLAUSE START ===") if c.strip()
            ]

    # Robust baseline fallback if multi-PDF text extractions are entirely missing from working disk
    if not chunks:
        chunks = [
            "[PO-101] General Examination Regulations Framework: Registration rules require valid fees paid prior to semester deadlines.",
            "[PO-201] Grading Matrix and Progress Limits: Cumulative grades over 3.5 flag immediate academic monitoring.",
            "[PO-301] Minimum Credit Point Sequence: Students must acquire at least 15 ECTS points per academic semester block.",
            "[PO-302] Examination repetition limits: Students are granted up to three attempts for mandatory core module examinations before termination.",
        ]

    # Fit the text standardizers for search vocabulary mapping
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunks)

    return model, scaler, kmeans, scaler_clustering, chunks, vectorizer, tfidf_matrix


# --- UNPACK MASTER RESERIALIZATION PLATFORM ---
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

# Establish layout partitioning instantly prior to capturing state updates
col1, col2 = st.columns([1, 1.2])

# ===========================================================================
# ADVISOR REGISTRATION INPUT PANEL (COL1)
# ===========================================================================
with col1:
    st.header("📋 Advisor Input Panel")
    with st.form(key=f"input_form_{st.session_state.form_key}"):

        # --- INSTRUCTOR SUGGESTION: Socio-economic Indicators Column ---
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

    # --- REACTIVE MATRIX CALCULATOR TRIGGER ---
    if submit_btn:
        # Move the cache clear step HERE so it only triggers precisely upon clicking submit
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

        # Structuring query row array mapping back features matching model parameters
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
                # 1. Transform inputs using RobustScaler for the main classification probability
                input_df_rf = input_df[scaler.feature_names_in_]
                scaled_rf = scaler.transform(input_df_rf)
                st.session_state.risk_pct = model.predict_proba(scaled_rf)[0][1] * 100

                # 2. Transform inputs using the dedicated StandardScaler for the K-Means profile routing
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
            # Safe heuristics baseline fallback if artifacts are empty or corrupted
            st.session_state.risk_pct = None
            st.session_state.cluster_id = 1 if (ects_s1 + ects_s2) < 30 else 0

        st.rerun()

    # --- AD-HOC CONSULTATION CONSOLE INJECTED LOWER LEFT ---
    st.markdown("---")
    st.markdown("### 💬 Ad-Hoc Regulatory Consultation Sandbox")
    st.markdown(
        "*Ask an ad-hoc, free-text question regarding specific regulations or edge cases for this student:*"
    )

    custom_question = st.text_input(
        "Enter your specific advisor question here:", key="advisor_free_question"
    )
    ask_button = st.button("💬 Query Examination Database")

    if ask_button and custom_question:
        if not GROQ_API_KEY:
            st.error(
                "❌ **LLM Configuration Error:** Missing `GROQ_API_KEY` system definition token inside environment keys."
            )
        else:
            c = (
                st.session_state.cached_student
                if st.session_state.cached_student
                else {
                    "residency": "Unspecified",
                    "student_job": "Unspecified",
                    "grade_s1": 1.0,
                    "grade_s2": 1.0,
                    "ects_s1": 30,
                    "ects_s2": 30,
                }
            )

            client = Groq(api_key=GROQ_API_KEY)
            question_vector = vectorizer.transform([custom_question])
            question_scores = cosine_similarity(question_vector, tfidf_matrix).flatten()

            st.session_state.sandbox_chunks = [
                chunks[idx][:1500] + "..."
                for idx in np.argsort(question_scores)[::-1][:3]
                if question_scores[idx] >= 0.00
            ]

            if st.session_state.sandbox_chunks:
                payload = "\n\n".join(st.session_state.sandbox_chunks)
                prompt = f"Student Profile: {c}. Regulation context: {payload}. Answer the following specific query: {custom_question}"

                with st.spinner("Synthesizing advice..."):
                    res = client.chat.completions.create(
                        model="llama-3.1-8b-instant",  # Safe token-saver model setup
                        messages=[{"role": "user", "content": prompt}],
                    )
                    st.session_state.sandbox_response = res.choices[0].message.content

    if st.session_state.sandbox_response is not None:
        st.markdown("---")
        st.success("#### 📋 Custom Consultation Answer")
        st.write(st.session_state.sandbox_response)

# ===========================================================================
# LIVE RETENTION INTEGRITY ANALYTICS CORE (COL2)
# ===========================================================================
with col2:
    st.header("⚡ Live Analytics Engine")
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

        # --- DYNAMIC HEURISTIC PROBABILITY CALIBRATION LAYER ---
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

        # --- DOMAIN-EXPERT HARD OVERRIDE LAYER ---
        # Safeguards top-performing profiles from outlier model inflation
        if c["grade_s1"] <= 2.0 and c["grade_s2"] <= 2.0:
            final_risk_pct = min(final_risk_pct, 25.0)

        # Render Alert Interfaces dynamically tracking our mathematical gradients
        if final_risk_pct >= 45.0:
            st.error(
                f"### ⚠️ HIGH RETENTION ALERT: **{final_risk_pct:.1f}% Attrition Probability** (Threshold: 45.0%)"
            )
        else:
            st.success(
                f"### ### ✅ **Stable Standing Profile: {final_risk_pct:.1f}% Attrition Probability**"
            )

        st.markdown("#### 📊 Risk Driver Deconstruction")
        st.columns(2)
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
            f"**👥Cohort Profile Focus:** {cluster_labels.get(cluster_id, 'Specialized Framework Segment Overview')}"
        )

        # ===========================================================================
        # VECTOR-MATCHED RAG ADVISORY GENERATION PLATFORM
        # ===========================================================================
        st.markdown("---")
        st.markdown("### 📋 Vector-Matched Examination Regulations (Prüfungsordnung)")

        queries = []
        if "Job" in c["student_job"]:
            queries.append(
                "fees tuition unpaid arrears payment deadline part time extension"
            )
        if c["ects_s1"] < 15 or c["ects_s2"] < 15:
            queries.append(
                "failed exam credit point minimum threshold or losing examination rights progress limits"
            )
        if c["grade_s1"] > 3.5 or c["grade_s2"] > 3.5:
            queries.append(
                "failed attempt repetition of examination grading scale fail attempts"
            )

        query_text = (
            " ".join([q for q in queries if q])
            or "standard admission requirements standing extension regulations"
        )
        scores = cosine_similarity(
            vectorizer.transform([query_text]), tfidf_matrix
        ).flatten()
        top_idx = np.argsort(scores)[::-1]

        matched_rules = [
            chunks[i][:1500] + "..." if len(chunks[i]) > 1500 else chunks[i]
            for i in top_idx[:2]
            if scores[i] >= 0.05
        ]

        if matched_rules and GROQ_API_KEY:
            client = Groq(api_key=GROQ_API_KEY)
            context_payload = "\n\n".join(matched_rules)

            s1_status = (
                "PASSING"
                if c["grade_s1"] <= 4.0
                else "CRITICAL MODULE FAIL (Academic Emergency)"
            )
            s2_status = (
                "PASSING"
                if c["grade_s2"] <= 4.0
                else "CRITICAL MODULE FAIL (Academic Emergency)"
            )
            trend_status = (
                "STABLE / IMPROVING MARKS"
                if c["grade_s2"] <= c["grade_s1"]
                else "WORSENING GRADIENT DIRECTION"
            )

            system_message = (
                "You are an expert academic advisor specialized in German university examination rules.\n\n"
                "STRICT ADVISORY REPORTING RULES:\n"
                "1. State exact qualitative definitions for Sem 1 and Sem 2 separately.\n"
                "2. Note that 60 ECTS means 1 year of progress completed.\n"
                "3. If grade number increases, sound a warning flag of performance decline.\n"
                "4. Use scale: 1.0-1.5 Sehr Gut, 1.6-2.5 Gut, 2.6-3.5 Befriedigend, 3.6-4.0 Ausreichend, >4.0 Nicht ausreichend.\n\n"
                "CRITICAL OUTPUT STRUCTURE DIRECTIVE:\nDo not write introductions. Output matching this exact markdown format:\n\n"
                "### 📋 Academic Advisory Assessment Report\n\n"
                "**Academic Standing**\n- [Insert classifications]\n\n"
                "**Trend Analysis**\n- [Insert trend breakdown]\n\n"
                "**Regulatory Directives**\n- [Insert actionable advice]"
            )

            user_message = f"""
<STUDENT_METRICS>
- Student Classification: {c['residency']}
- Employment Job Status: {c['student_job']}
- Total Combined Earned ECTS: {c['ects_s1'] + c['ects_s2']} points
- Semester 1 Numeric Grade: {c['grade_s1']} -> Evaluated Stand: {s1_status}
- Semester 2 Numeric Grade: {c['grade_s2']} -> Evaluated Stand: {s2_status}
- Performance Trend Direction: {trend_status}
- Pipeline Attrition Score Value: {final_risk_pct:.1f}% Attrition Risk Score
</STUDENT_METRICS>

<REGULATORY_CONTEXT_BLOCKS>
{context_payload}
</REGULATORY_CONTEXT_BLOCKS>
"""
            try:
                with st.spinner(
                    "LLM synthesizing verified student regulatory advice..."
                ):
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",  # Token-saver setup active
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message},
                        ],
                        temperature=0.0,
                    )
                    st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(
                    f"❌ **Groq Authentication Crash during RAG Generation:** {str(e)}"
                )

        with st.expander("🔎 View Source Clauses [PO-101]"):
            for i, rule in enumerate(matched_rules, 1):
                st.info(f"**Source Context Block #{i}:**\n{rule}")
    else:
        st.info(
            "ℹ️ Fill out student parameters on the left panel and click **Run Prediction & RAG Analysis**."
        )
