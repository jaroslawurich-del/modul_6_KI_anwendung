import streamlit as st
from agent import agent

def chat():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        st.chat_message(m["role"]).markdown(m["content"])

    q = st.chat_input("Frage")

    if q:

        st.session_state.messages.append({"role": "user", "content": q})

        res = agent.invoke({"question": q})

        answer = res["answer"]

        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.rerun()