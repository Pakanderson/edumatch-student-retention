import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from utils import (
    apply_custom_styles,
    init_session_state,
    load_all_assets,
    GROQ_API_KEY,
)

# ---------------------------------------------------------------------------
# 1. Configuration (CRITICAL: MUST BE THE FIRST STREAMLIT COMMAND)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="POLICY REGULATIONS AND ADVISORY DASHBOARD",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# SLIDE 2: Mid-point Transition
# (This will be the first thing the panel sees when you click Page 2)
# ---------------------------------------------------------------------------
st.image("assets/mid_slide.jpg", use_container_width=True)


# ---------------------------------------------------------------------------
# 2. System Initialization & Data Loading
# ---------------------------------------------------------------------------
apply_custom_styles()
init_session_state()

model, scaler, kmeans, scaler_clustering, chunks, vectorizer, tfidf_matrix = (
    load_all_assets()
)

st.title("⚖️ POLICY REGULATIONS AND ADVISORY DASHBOARD")
st.markdown("---")

if st.session_state.cached_student is None:
    st.warning(
        "⚠️ No active student profile selected. Please enter metrics on **Page 1 (Student Profile & LIVE ANALYTICS)** first."
    )
else:
    c = st.session_state.cached_student

    # Summary Context Card
    st.markdown("### 📌 PROFILE SUMMARY")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Degree Track", c["is_master"].split()[0])
    sc2.metric("Sem 1 Grade / ECTS", f"{c['grade_s1']} / {c['ects_s1']}")
    sc3.metric("Sem 2 Grade / ECTS", f"{c['grade_s2']} / {c['ects_s2']}")
    sc4.metric(
        "Residency Status", "Non-EU" if "Non-EU" in c["residency"] else "EU/Domestic"
    )

    st.markdown("---")

    tab1, tab2 = st.tabs(
        [
            "📋 Automated Policy Advisory Report",
            "💬 Ad-Hoc Regulatory Consultation Sandbox",
        ]
    )

    # TAB 1: RAG REPORT GENERATOR
    with tab1:
        st.subheader("📋 Vector-Matched Examination Regulations (Prüfungsordnung)")

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
</STUDENT_METRICS>

<REGULATORY_CONTEXT_BLOCKS>
{context_payload}
</REGULATORY_CONTEXT_BLOCKS>
"""
            if st.button("⚡ VIEW FULL REGULATORY REPORT", use_container_width=True):
                try:
                    with st.spinner(
                        "LLM synthesizing verified student regulatory advice..."
                    ):
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
                        f"❌ **Groq Authentication Crash during RAG Generation:** {str(e)}"
                    )

        with st.expander("🔎 View Matched Source Regulatory Clauses"):
            for i, rule in enumerate(matched_rules, 1):
                st.info(f"**Source Context Block #{i}:**\n{rule}")

    # TAB 2: AD-HOC CONSULTATION SANDBOX
    with tab2:
        st.subheader("💬 Ad-Hoc Regulatory Consultation Sandbox")
        st.markdown(
            "*Ask a free-text question regarding specific regulations or edge cases for this student:*"
        )

        custom_question = st.text_input(
            "Enter your specific advisor question here:", key="advisor_free_question"
        )
        ask_button = st.button("💬 Query Examination Database")

        if ask_button and custom_question:
            if not GROQ_API_KEY:
                st.error(
                    "❌ **LLM Configuration Error:** Missing `GROQ_API_KEY` environment token."
                )
            else:
                client = Groq(api_key=GROQ_API_KEY)
                question_vector = vectorizer.transform([custom_question])
                question_scores = cosine_similarity(
                    question_vector, tfidf_matrix
                ).flatten()

                st.session_state.sandbox_chunks = [
                    chunks[idx][:1500] + "..."
                    for idx in np.argsort(question_scores)[::-1][:3]
                    if question_scores[idx] >= 0.00
                ]

                if st.session_state.sandbox_chunks:
                    payload = "\n\n".join(st.session_state.sandbox_chunks)
                    prompt = f"Student Profile: {c}. Regulation context: {payload}. Answer the following query: {custom_question}"

                    with st.spinner("Synthesizing advice..."):
                        res = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": prompt}],
                        )
                        st.session_state.sandbox_response = res.choices[
                            0
                        ].message.content

        if st.session_state.sandbox_response is not None:
            st.markdown("---")
            st.success("#### 📋 Custom Consultation Answer")
            st.write(st.session_state.sandbox_response)

# ---------------------------------------------------------------------------
# SLIDE 3: Closing Slide (Displays at the very bottom of the page)
# ---------------------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.image("assets/final_slide.jpg", use_container_width=True)
