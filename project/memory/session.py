import streamlit as st
from memory.conversation import Conversation

def get_session():
    if "conv" not in st.session_state:
        st.session_state.conv = Conversation()
    return st.session_state.conv