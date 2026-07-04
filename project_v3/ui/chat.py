import streamlit as st

from agent import agent

from config.settings import TEMPERATURE


def chat_ui():

    st.subheader("Chat")

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(TEMPERATURE),
        step=0.1
    )

    question = st.chat_input(
        "Frage stellen..."
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Antwort wird erzeugt..."):

            result = agent.invoke(
                {
                    "question": question,
                    "temperature": temperature,
                }
            )

        with st.chat_message("assistant"):
            st.write(result["answer"])