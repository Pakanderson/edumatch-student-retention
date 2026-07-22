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
        str(raw_key)
        .strip()
        .replace('"', "")
        .replace("'", "")
        .replace("\r", "")
        .replace("\n", "")
    )


def apply_custom_styles():
    """Injects compact presentation CSS and specific button color overrides."""
    st.markdown(
        """
        <style>
            /* 1. Safe top padding to prevent title cut-off under Streamlit header */
            .block-container {
                padding-top: 2.2rem !important;
                padding-bottom: 0.8rem !important;
                padding-left: 1.8rem !important;
                padding-right: 1.8rem !important;
            }

            /* 2. Compact headers and title typography */
            h1 {
                text-align: center !important;
                font-size: 1.85rem !important;
                padding-bottom: 0.1rem !important;
                margin-top: 0.0rem !important;
                margin-bottom: 0.4rem !important;
            }
            h2 {
                font-size: 1.25rem !important;
                margin-top: 0.1rem !important;
                margin-bottom: 0.3rem !important;
            }
            h3, h4 {
                font-size: 1.05rem !important;
                font-weight: bold !important;
                margin-top: 0.1rem !important;
                margin-bottom: 0.2rem !important;
            }

            /* 3. Base text and widget label scaling */
            html, body, [data-testid="stWidgetLabel"] p, .stSelectbox div, .stMarkdown p {
                font-size: 0.85rem !important;
            }

            /* 4. Form container & widget spacing */
            div[data-testid="stForm"] {
                padding: 0.6rem 0.8rem !important;
                margin-bottom: 0.4rem !important;
            }

            .stSelectbox, .stSlider, .stNumberInput {
                margin-bottom: -0.6rem !important;
            }

            div[data-baseweb="select"] > div {
                min-height: 2.0rem !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
            }

            /* 5. Scaled submit button (Run Prediction) */
            div[data-testid="stForm"] div[data-testid="column"]:nth-of-type(1) button {
                background-color: #2e7d32 !important; 
                color: white !important;              
                font-size: 0.95rem !important;
                font-weight: bold !important;
                height: 2.2em !important;
                width: 100% !important;
                border-radius: 6px !important;
                border: none !important;
            }

            /* 6. Scaled Clear button (Red) */
            div[data-testid="stForm"] div[data-testid="column"]:nth-of-type(2) button {
                background-color: #ff4b4b !important;
                color: white !important;
                font-size: 0.95rem !important;
                font-weight: bold !important;
                height: 2.2em !important;
                width: 100% !important;
                border-radius: 6px !important;
                border: none !important;
            }

            /* 7. Metric Card Display */
            [data-testid="stMetricValue"] {
                font-size: 1.4rem !important;
            }
            [data-testid="stMetricLabel"] p {
                font-size: 0.80rem !important;
            }
            
            hr {
                margin: 0.4rem 0 !important;
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
        pass

    chunks = []
    if os.path.exists("raw_extracted_po.txt"):
        with open("raw_extracted_po.txt", "r", encoding="utf-8") as f:
            chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]

    if not chunks:
        chunks = ["[PO-101] General Regs.", "[PO-302] Hardship Apps."]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunks)

    return model, scaler, kmeans, scaler_clustering, chunks, vectorizer, tfidf_matrix


CLUSTER_LABELS = {
    0: "Academic Progress High Risk",
    1: "Moderate Credit Risk",
    # ... (other clusters)
}
