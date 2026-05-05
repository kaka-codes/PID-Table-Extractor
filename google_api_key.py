import os

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None


def _load_google_api_key() -> str:
    if st is not None:
        try:
            secret_value = str(st.secrets["GOOGLE_API_KEY"]).strip()
            if secret_value:
                return secret_value
        except Exception:
            pass

    return str(os.getenv("GOOGLE_API_KEY", "")).strip()


google_api_key = _load_google_api_key()
