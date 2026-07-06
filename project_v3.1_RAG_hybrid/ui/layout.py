import streamlit as st

from ui.status import status_ui
from ui.upload import upload_ui
from ui.chat import chat_ui


def run_app():

    st.set_page_config(
        page_title="KI Dokumentenassistent",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 KI Dokumentenassistent")

    # -----------------------------
    # Session State Safety Init
    # -----------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False

    # -----------------------------
    # STATUS SECTION
    # -----------------------------
    with st.container():
        try:
            status_ui()
        except Exception as e:
            st.error(f"Status UI Fehler: {e}")

    st.divider()

    # -----------------------------
    # UPLOAD SECTION
    # -----------------------------
    with st.container():
        try:
            upload_ui()
        except Exception as e:
            st.error(f"Upload UI Fehler: {e}")

    st.divider()

    # -----------------------------
    # CHAT SECTION
    # -----------------------------
    with st.container():
        try:
            chat_ui()
        except Exception as e:
            st.error(f"Chat UI Fehler: {e}")