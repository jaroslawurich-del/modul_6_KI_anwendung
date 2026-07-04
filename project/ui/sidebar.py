import streamlit as st

def sidebar():
    with st.sidebar:
        st.header("Einstellungen")
        st.slider("Temperature", 0.0, 1.0, 0.0)