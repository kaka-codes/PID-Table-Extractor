import streamlit as st


try:
    google_api_key = str(st.secrets["GOOGLE_API_KEY"]).strip()
except Exception:
    google_api_key = ""
