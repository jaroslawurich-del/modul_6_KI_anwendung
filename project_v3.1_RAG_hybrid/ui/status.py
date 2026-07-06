import streamlit as st

from llm.health import check_ollama


def status_ui():

    if check_ollama():
        st.sidebar.success("🟢 Ollama erreichbar")
    else:
        st.sidebar.error("🔴 Ollama nicht erreichbar")