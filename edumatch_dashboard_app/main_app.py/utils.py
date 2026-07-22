import streamlit as st
import streamlit as st
import joblib


# 1. Styling function
def apply_custom_styles():
    st.markdown(
        """
        <style>
            h1 { text-align: center !important; font-size: 3.5rem !important; }
            html, body, .stMarkdown p { font-size: 1.25rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# 2. Asset loading function
@st.cache_resource
def load_all_assets():
    model = joblib.load("models/german_retention_model.pkl")
    scaler = joblib.load("models/clustering_scaler.pkl")
    kmeans = joblib.load("models/kmeans_model.pkl")
    # ... any other assets ...
    return model, scaler, kmeans


def apply_custom_styles():
    """Injects centralized CSS for consistent font sizing and styling."""
    st.markdown(
        """
        <style>
            /* 1. Center the master application title */
            h1 {
                text-align: center !important;
                font-size: 3.5rem !important;
                padding-bottom: 1rem !important;
            }
            
            /* 2. Global presentation font scaling for compact viewing */
            html, body, [data-testid="stWidgetLabel"] p, .stSelectbox div, .stMarkdown p {
                font-size: 1.25rem !important;
            }
            
            /* 3. Subheader emphasis */
            h3, h4 {
                font-size: 1.5rem !important;
                font-weight: bold !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
