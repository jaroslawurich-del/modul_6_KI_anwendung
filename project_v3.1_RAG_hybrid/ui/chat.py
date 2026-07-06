import streamlit as st

from agent.nodes import rag_node


def chat_ui():

    st.subheader("Chat 🤖")

    # -----------------------------
    # MEMORY INIT
    # -----------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # -----------------------------
    # SHOW HISTORY
    # -----------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # -----------------------------
    # INPUT
    # -----------------------------
    question = st.chat_input("Frage stellen...")

    if question:

        # User message speichern
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.write(question)

        # -----------------------------
        # RAG PROCESSING
        # -----------------------------
        with st.spinner("Denke nach... 🤔"):

            result = rag_node(
                {
                    "question": question,
                    "answer": ""
                }
            )

            answer = result["answer"]

        # Assistant message speichern
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)