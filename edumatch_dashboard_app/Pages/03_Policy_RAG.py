import streamlit as st
from groq import Groq
import streamlit as st
from utils import apply_custom_styles
import streamlit as st
from utils import apply_custom_styles, load_all_assets

# 1. Apply styles immediately
apply_custom_styles()

# 2. Load assets (this will only run once and then use the cache)
model, scaler, kmeans = load_all_assets()

# Call this immediately after st.set_page_config
apply_custom_styles()

st.title("🎓 EduMatch Dashboard")
# The rest of your page code...

st.set_page_config(page_title="Policy Brief", layout="wide")

st.title("⚖️ Regulatory RAG Brief")

if st.session_state.student_data:
    # 1. Retrieve RAG chunks (use the logic from your original code)
    # 2. Call Groq LLM
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    # ... add your generation logic ...
    st.subheader("Intervention Directive")
    st.markdown("LLM response goes here with enlarged fonts...")
else:
    st.info("Navigate to the Dashboard page to generate a policy brief.")
