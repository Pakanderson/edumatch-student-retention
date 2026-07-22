import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment secrets
load_dotenv(override=True)


def get_groq_api_key():
    raw_key = os.environ.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not raw_key and "GROQ_API_KEY" in st.secrets:
        raw_key = st.secrets["GROQ_API_KEY"]

    if raw_key:
        return (
            str(raw_key)
            .strip()
            .replace('"', "")
            .replace("'", "")
            .replace("\r", "")
            .replace("\n", "")
        )
    return None


def apply_custom_styles():
    """Injects high-impact presentation typography and visual styling across all pages."""
    st.markdown(
        """
        <style>
            /* 1. Master Title styling */
            h1 {
                text-align: center !important;
                font-size: 3.5rem !important;
                font-weight: 800 !important;
                padding-bottom: 1rem !important;
            }
            
            /* 2. Scaled presentation typography */
            html, body, [data-testid="stWidgetLabel"] p, .stSelectbox div, .stMarkdown p {
                font-size: 1.35rem !important;
                line-height: 1.6 !important;
            }
            
            /* 3. Section subheaders */
            h2, h3, h4 {
                font-weight: 700 !important;
            }

            /* 4. Form Submit Button styling */
            .stFormSubmitButton > button {
                background-color: #2e7d32 !important; 
                color: white !important;              
                font-size: 1.35rem !important;
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


@st.cache_resource(show_spinner=False)
def load_all_assets():
    """Loads ML artifacts, scalers, and fits TF-IDF text vectorizers once into memory."""
    model, scaler, kmeans, scaler_clustering = None, None, None, None
    try:
        model = joblib.load("models/german_retention_model.pkl")
        scaler = joblib.load("models/clustering scaler.pkl")
        kmeans = joblib.load("models/kmeans_model.pkl")
        scaler_clustering = joblib.load("models/clustering_scaler.pkl")
    except Exception as e:
        st.error(f"❌ Error loading system files from models/ directory: {e}")

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


def init_session_state():
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
