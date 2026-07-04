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

    status_ui()

    upload_ui()

    st.divider()

    chat_ui()