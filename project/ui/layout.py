import streamlit as st
from ui.chat import chat
from ui.upload import upload
from ui.status import status

def run_app():

    st.set_page_config("KI Assistent")

    status()
    upload()
    chat()