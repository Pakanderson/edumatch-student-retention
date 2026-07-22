import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from utils import (
    apply_custom_styles,
    load_all_assets,
    get_groq_api_key,
    init_session_state,
)

st.set_page_config(page_title="Policy RAG & Briefing", page_icon="📋", layout="wide")

apply_custom_styles()
init_session_state()

model, scaler, kmeans, scaler_clustering, chunks, vectorizer, tfidf_matrix = (
    load_all_assets()
)
GROQ_API_KEY = get_groq_api_key()

st.title("📋 Policy RAG & Advisory Guidance")

# --- MOVED SANDBOX TO TOP ---
st.markdown("### 💬 Ad-Hoc Regulatory Consultation Sandbox")
st.info("Ask specific questions about the University Examination Regulations here.")
custom_question = st.text_input(
    "Enter your specific advisor question:", key="advisor_free_question"
)
ask_button = st.button("💬 Query Examination Database")

if ask_button and custom_question:
    if not GROQ_API_KEY:
        st.error("❌ LLM Error: Missing API Key.")
    else:
        client = Groq(api_key=GROQ_API_KEY)
        question_vector = vectorizer.transform([custom_question])
        question_scores = cosine_similarity(question_vector, tfidf_matrix).flatten()

        # Retrieve context
        sandbox_chunks = [
            chunks[idx][:1500] + "..."
            for idx in np.argsort(question_scores)[::-1][:3]
            if question_scores[idx] >= 0.05
        ]

        if sandbox_chunks:
            payload = "\n\n".join(sandbox_chunks)
            prompt = f"Use this regulation context to answer: {custom_question}\n\nContext: {payload}"

            with st.spinner("Synthesizing advice..."):
                res = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                )
                st.session_state.sandbox_response = res.choices[0].message.content
        else:
            st.warning("No relevant regulatory clauses found for this query.")

if st.session_state.sandbox_response:
    st.success("#### 📋 Custom Consultation Answer")
    st.write(st.session_state.sandbox_response)

st.markdown("---")
# --- AUTOMATED BRIEFING SECTION ---
st.markdown("### 🤖 Automated Advisor Briefing")
if st.session_state.cached_student is not None:
    # ... (existing briefing code) ...
    st.write("Briefing based on student risk profile generated below...")
else:
    st.info(
        "Fill out student parameters on the main page to generate the automated briefing."
    )
