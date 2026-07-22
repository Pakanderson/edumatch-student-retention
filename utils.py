import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# --- INITIALIZE ENVIRONMENT ---
load_dotenv(override=True)

# Bulk-strip trailing spaces or artifacts from API Key
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


def apply_custom_styles():
    """Injects compact presentation CSS to minimize page scrolling."""
    st.markdown(
        """
        <style>
            /* 1. Reduce top and bottom whitespace on main page container */
            .block-container {
                padding-top: 1.2rem !important;
                padding-bottom: 1rem !important;
                padding-left: 2rem !important;
                padding-right: 2rem !important;
            }

            /* 2. Scaled title and headers */
            h1 {
                text-align: center !important;
                font-size: 2.1rem !important;
                padding-bottom: 0.2rem !important;
                margin-bottom: 0.5rem !important;
            }
            h2 {
                font-size: 1.4rem !important;
                margin-top: 0.2rem !important;
                margin-bottom: 0.4rem !important;
            }
            h3, h4 {
                font-size: 1.15rem !important;
                font-weight: bold !important;
                margin-top: 0.2rem !important;
                margin-bottom: 0.3rem !important;
            }

            /* 3. Compact text sizing for descriptions, paragraphs, and labels */
            html, body, [data-testid="stWidgetLabel"] p, .stSelectbox div, .stMarkdown p {
                font-size: 0.92rem !important;
            }

            /* 4. Tighten widget spacing and margins */
            div[data-testid="stForm"] {
                padding: 0.8rem 1rem !important;
                margin-bottom: 0.5rem !important;
            }

            .stSelectbox, .stSlider, .stNumberInput {
                margin-bottom: -0.4rem !important;
            }

            div[data-baseweb="select"] > div {
                min-height: 2.2rem !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
            }

            /* 5. Scaled and compact submit button */
            .stFormSubmitButton > button {
                background-color: #2e7d32 !important; 
                color: white !important;              
                font-size: 1.05rem !important;
                font-weight: bold !important;
                height: 2.4em !important;
                width: 100% !important;
                border-radius: 6px !important;
                border: none !important;
                margin-top: 0.5rem !important;
            }
            .stFormSubmitButton > button:hover {
                background-color: #1b5e20 !important; 
                color: #ffffff !important;
            }

            /* 6. Compact Metric Cards */
            [data-testid="stMetricValue"] {
                font-size: 1.6rem !important;
            }
            [data-testid="stMetricLabel"] p {
                font-size: 0.88rem !important;
            }

            /* 7. Alert containers compact padding */
            div[data-testid="stAlert"] {
                padding: 0.6rem 0.8rem !important;
                margin-bottom: 0.5rem !important;
            }
            
            hr {
                margin: 0.6rem 0 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state():
    """Initializes global session state values across pages."""
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
    """Resets input session data."""
    st.session_state.form_key += 1
    st.session_state.cached_student = None
    st.session_state.risk_pct = None
    st.session_state.cluster_id = None
    st.session_state.sandbox_response = None
    st.session_state.sandbox_chunks = None


@st.cache_resource(show_spinner=False)
def load_all_assets():
    """Loads machine learning models, scalers, and TF-IDF text vectorizers."""
    model, scaler, kmeans, scaler_clustering = None, None, None, None
    try:
        model = joblib.load("models/german_retention_model.pkl")
        scaler = joblib.load("models/clustering scaler.pkl")
        kmeans = joblib.load("models/kmeans_model.pkl")
        scaler_clustering = joblib.load("models/clustering_scaler.pkl")
    except Exception:
        pass  # Handled safely by fallbacks in main runtime

    # Regulatory Text Ingestion
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

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunks)

    return model, scaler, kmeans, scaler_clustering, chunks, vectorizer, tfidf_matrix


CLUSTER_LABELS = {
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
