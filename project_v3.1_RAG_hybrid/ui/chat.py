import streamlit as st

from agent.nodes import rag_node
from config.settings import TEMPERATURE


def chat_ui():

    st.subheader("💬 Chat")

    # -----------------------------
    # Sidebar
    # -----------------------------
    st.sidebar.header("⚙️ Einstellungen")

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("temperature", TEMPERATURE)),
        step=0.1,
        help="Niedrig = präziser, Hoch = kreativer"
    )

    st.session_state.temperature = temperature

    # -----------------------------
    # Chat-Historie
    # -----------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # -----------------------------
    # Eingabe
    # -----------------------------
    question = st.chat_input("Frage stellen...")

    if not question:
        return

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Antwort wird erzeugt..."):

        result = rag_node(
            {
                "question": question,
                "temperature": temperature,
            }
        )

    answer = result["answer"]

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        #st.markdown(answer)
        st.write(answer)