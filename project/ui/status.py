import streamlit as st
from llm.health import check_ollama

def status():

    if check_ollama():
        st.success("🟢 Ollama OK")
    else:
        st.error("🔴 Ollama nicht erreichbar")