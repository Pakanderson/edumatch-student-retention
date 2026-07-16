import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===========================================================================
# 1. STRUCTURAL INITIALIZATION, SECURITY & GLOBAL STYLES
# ===========================================================================

# Enforce app-wide layout grid spacing constraints immediately
st.set_page_config(
    page_title="EduMatch Academic Advisory Suite", page_icon="🎓", layout="wide"
)

# Target-locked global CSS injection to ensure text prominence
st.markdown(
    """
<style>
    div.stButton > button:first-child {
        font-size: 1.25rem !important;
        font-weight: bold !important;
        height: 3em !important;
        width: 100% !important;
    }
    .stFormSubmitButton > button {
        font-size: 1.25rem !important;
        font-weight: bold !important;
        background-color: #ff4b4b !important;
        color: white !important;
        height: 3em !important;
        width: 100% !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Securely extract and clean production system tokens
raw_key = (
    st.secrets.get("GROQ_API_KEY")
    if "GROQ_API_KEY" in st.secrets
    else os.environ.get("GROQ_API_KEY")
)
GROQ_API_KEY = None
if raw_key:
    GROQ_API_KEY = (
        str(raw_key)
        .strip()
        .replace('"', "")
        .replace("'", "")
        .replace("\r", "")
        .replace("\n", "")
    )

# ===========================================================================
# 2. ASSET PACK LOADING & RAG ENGINE CACHING
# ===========================================================================


@st.cache_resource(show_spinner=False)
def load_all_assets():
    """
    Loads machine learning pipeline artifacts using verified files on disk.
    Automatically initializes the baseline TF-IDF matrix for the text scanner.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load ML Classifier and Cluster Components
    try:
        model = joblib.load(
            os.path.join(base_dir, "models", "german_retention_model.pkl")
        )
        kmeans = joblib.load(os.path.join(base_dir, "models", "kmeans_model.pkl"))
        scaler_clustering = joblib.load(
            os.path.join(base_dir, "models", "clustering_scaler.pkl")
        )
    except Exception as e:
        st.error(f"❌ Error loading production system files: {e}")
        model, kmeans, scaler_clustering = None, None, None

    # 2. Ingest Regulatory Framework Text Assets
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
            "[PO-101] General Examination Regulations Framework: Registration rules require valid fees paid prior to semester deadlines.",
            "[PO-201] Grading Matrix and Progress Limits: Cumulative grades over 3.5 flag immediate academic monitoring.",
            "[PO-301] Minimum Credit Point Sequence: Students must acquire at least 15 ECTS points per academic semester block.",
            "[PO-302] Examination repetition limits: Students are granted up to three attempts for mandatory core module examinations before termination.",
        ]

    # 3. Vectorize Knowledge Base Blocks
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunks)

    return model, kmeans, scaler_clustering, chunks, vectorizer, tfidf_matrix


# Execute asset assembly load pass
model, kmeans, scaler_clustering, chunks, vectorizer, tfidf_matrix = load_all_assets()

# ===========================================================================
# 3. STATE SESSION ARCHITECTURE & CLEANERS
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
if "academic_score" not in st.session_state:
    st.session_state.academic_score = 0.0
if "socio_score" not in st.session_state:
    st.session_state.socio_score = 0.0


def clear_inputs():
    st.session_state.form_key += 1
    st.session_state.cached_student = None
    st.session_state.risk_pct = None
    st.session_state.cluster_id = None
    st.session_state.sandbox_response = None
    st.session_state.sandbox_chunks = None
    st.session_state.academic_score = 0.0
    st.session_state.socio_score = 0.0


# ===========================================================================
# 4. USER INTERACTION INTERFACE & ADVISOR INPUT PANEL (COL1)
# ===========================================================================

st.title("🎓 EduMatch: Predictive Student Retention and Prescriptive Analytics")
col1, col2 = st.columns([1.1, 1.2], gap="large")

with col1:
    st.header("📋 Advisor Input Panel")

    # Establish a single unified transactional form element block
    with st.form(key=f"input_form_{st.session_state.form_key}"):

        st.subheader("🌍 Socio-economic Indicators")
        gender = st.selectbox("Gender", ["Female", "Male"])
        is_master = st.selectbox(
            "Enrolled Degree Level",
            ["Bachelor Level Degree Program", "Master Level Degree program"],
        )
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

        st.markdown("---")

        st.subheader("Academic Milestones")
        ects_s1 = st.number_input("ECTS Credits Earned (Sem 1)", 0, 40, 12)
        grade_s1 = st.slider(
            "Grade Average (Sem 1) [1.0 Best to 5.0 Fail]", 1.0, 5.0, 3.8, 0.1
        )
        ects_s2 = st.number_input("ECTS Credits Earned (Sem 2)", 0, 40, 10)
        grade_s2 = st.slider(
            "Grade Average (Sem 2) [1.0 Best to 5.0 Fail]", 1.0, 5.0, 3.9, 0.1
        )

        submit_btn = st.form_submit_button("🚀 Run Prediction & RAG Analysis")

    # Layout reset command immediately below form structure
    if st.button("🧹 Clear All Advisor Inputs", use_container_width=True):
        clear_inputs()
        st.rerun()

    # Dynamic Live Sub-Heuristics Evaluator Logic Step
    # Evaluates immediately to break any frozen calculations
    academic_calc = (
        50.0 if (ects_s1 + ects_s2) < 25 else (20.0 + (max(grade_s1, grade_s2) * 5.0))
    )
    socio_calc = 15.0
    if "Part-Time" in student_job:
        socio_calc += 20.0
    if "Non-EU" in residency:
        socio_calc += 20.0
    if "Yes (Recipient)" in bafoeg and (ects_s1 + ects_s2) < 30:
        socio_calc += 25.0

    academic_calc = min(95.0, max(5.0, academic_calc))
    socio_calc = min(95.0, max(5.0, socio_calc))

    # --- REACTIVE MATRIX CALCULATOR TRIGGER ---
    if submit_btn:
        st.session_state.academic_score = academic_calc
        st.session_state.socio_score = socio_calc

        # Prepare state map package for serialization
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

        # 1. Execute Inference against Random Forest Pipeline Classifier
        if model is not None:
            try:
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
                prob = model.predict_proba(input_df)[0][1] * 100.0
                st.session_state.risk_pct = prob
            except Exception:
                st.session_state.risk_pct = None
        else:
            st.session_state.risk_pct = None

        # 2. Execute Cohort Group Lookup against Scaled K-Means Cluster
        if kmeans is not None and scaler_clustering is not None:
            try:
                cluster_features = np.array([[ects_s1, grade_s1, ects_s2, grade_s2]])
                scaled_features = scaler_clustering.transform(cluster_features)
                st.session_state.cluster_id = int(kmeans.predict(scaled_features)[0])
            except Exception:
                st.session_state.cluster_id = 1 if (ects_s1 + ects_s2) < 30 else 0
        else:
            st.session_state.cluster_id = 1 if (ects_s1 + ects_s2) < 30 else 0

        st.rerun()

    # --- AD-HOC CONSULTATION CONSOLE INJECTED LOWER LEFT ---
    st.markdown("---")
    st.markdown("### 💬 Ad-Hoc Regulatory Consultation Sandbox")
    st.markdown("*Ask an ad-hoc compliance question based on the regulations loaded:*")

    custom_question = st.text_input(
        "Enter your compliance query:", key="advisor_free_question"
    )
    ask_button = st.button("💬 Query Regulations System")

    if ask_button and custom_question:
        if not GROQ_API_KEY:
            st.error("❌ LLM Configuration Error: Missing GROQ_API_KEY system token.")
        else:
            c = (
                st.session_state.cached_student
                if st.session_state.cached_student
                else {
                    "residency": residency,
                    "student_job": student_job,
                    "bafoeg": bafoeg,
                    "grade_s1": grade_s1,
                    "grade_s2": grade_s2,
                    "ects_s1": ects_s1,
                    "ects_s2": ects_s2,
                }
            )

            # Map query context blocks semantically
            question_vector = vectorizer.transform([custom_question])
            question_scores = cosine_similarity(question_vector, tfidf_matrix).flatten()
            st.session_state.sandbox_chunks = [
                chunks[idx][:1500] + "..."
                for idx in np.argsort(question_scores)[::-1][:3]
                if question_scores[idx] >= 0.00
            ]

            if st.session_state.sandbox_chunks:
                payload = "\n\n".join(st.session_state.sandbox_chunks)
                prompt = f"Student Profile: {c}. Regulation context: {payload}. Answer: {custom_question}"

                try:
                    client = Groq(api_key=GROQ_API_KEY)
                    # Switched to 8B token-saver model for bulletproof reliability
                    res = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    st.session_state.sandbox_response = res.choices[0].message.content
                except Exception as e:
                    st.error(f"❌ Sandbox Query Error: {e}")

    if st.session_state.sandbox_response is not None:
        st.success("#### 📋 Custom Consultation Answer")
        st.write(st.session_state.sandbox_response)

# ===========================================================================
# 5. LIVE RETENTION INTEGRITY ANALYTICS CORE & RAG (COL2)
# ===========================================================================

with col2:
    st.header("⚡ Live Analytics Engine")

    # 1. State Decoupling Evaluation
    if st.session_state.risk_pct is not None:
        final_risk_pct = st.session_state.risk_pct
    else:
        final_risk_pct = (academic_calc * 0.60) + (socio_calc * 0.40)

    final_risk_pct = min(98.5, max(4.5, final_risk_pct))

    # Enforce panel-approved 40.0% alert gradient conditions
    if final_risk_pct >= 40.0:
        st.error(
            f"### ⚠️ HIGH RETENTION ALERT: {final_risk_pct:.1f}% Attrition Probability (Threshold: 40.0%)"
        )
    else:
        st.success(
            f"### ✅ Stable Standing Profile: {final_risk_pct:.1f}% Attrition Probability"
        )

    # 2. Risk Metrics Breakdown Visualization Area
    st.markdown("#### 📊 Risk Driver Deconstruction")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        current_acad = (
            st.session_state.academic_score
            if st.session_state.cached_student
            else academic_calc
        )
        st.metric(label="📚 Operational Academic Risk", value=f"{current_acad:.1f}%")
        if current_acad > 45.0:
            st.markdown("🔴 *Severe credit/grade progression gaps.*")
        else:
            st.markdown("🟢 *Progress stable.*")
    with d_col2:
        current_socio = (
            st.session_state.socio_score
            if st.session_state.cached_student
            else socio_calc
        )
        st.metric(label="🌍 Socioeconomic Strain Score", value=f"{current_socio:.1f}%")
        if current_socio > 45.0:
            st.markdown("🔴 *High employment or structural friction.*")
        else:
            st.markdown("🟢 *Context clear.*")

    st.markdown("---")

    # 3. Dynamic Cluster Cohort Lookups
    cluster_id = (
        st.session_state.cluster_id
        if st.session_state.cluster_id is not None
        else (1 if (ects_s1 + ects_s2) < 25 else 0)
    )
    cluster_labels = {
        0: "Cluster 0: High Academic Progress with Structural Risk Factors",
        1: "Cluster 1: Moderate Credit Accumulation and Study-Load Risk",
        2: "Cluster 2: Early Non-Engagement Profile: Younger Male Students",
        3: "Cluster 3: Employed Student Study-Work Pressure Profile",
        4: "Cluster 4: International Student Transition and Credit-Progress Risk",
        5: "Cluster 5: BAföG Recipient Financial-Support and Progression Risk",
    }
    st.markdown(
        f"**👥 Assigned Support Intervention Cohort:** {cluster_labels.get(cluster_id, 'Specialized Framework Segment Overview')}"
    )

    # ===========================================================================
    # 6. VECTOR-MATCHED RAG ADVISORY GENERATION PLATFORM
    # ===========================================================================

    st.markdown("---")
    st.markdown("### 📋 Vector-Matched Examination Regulations (Prüfungsordnung)")

    # Base evaluations on live UI values if submission has not happened yet
    c = (
        st.session_state.cached_student
        if st.session_state.cached_student
        else {
            "residency": residency,
            "student_job": student_job,
            "bafoeg": bafoeg,
            "grade_s1": grade_s1,
            "grade_s2": grade_s2,
            "ects_s1": ects_s1,
            "ects_s2": ects_s2,
        }
    )

    if st.session_state.cached_student is not None:
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
                " Academic Standing \n- [Insert classifications]\n\n"
                " Trend Analysis \n- [Insert trend breakdown]\n\n"
                " Regulatory Directives \n- [Insert actionable advice]"
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
                    client = Groq(api_key=GROQ_API_KEY)
                    # Switched to 8B token-saver model for ultimate quota performance
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message},
                        ],
                        temperature=0.0,
                    )
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(
                    f"❌ Groq Authentication Crash during RAG Generation: {str(e)}"
                )

            with st.expander("🔎 View Source Clauses [PO-101]"):
                for i, rule in enumerate(matched_rules, 1):
                    st.info(f" Source Context Block #{i}: \n{rule}")
    else:
        st.info(
            "ℹ️ Fill out student parameters on the left panel and click Run Prediction & RAG Analysis ."
        )
