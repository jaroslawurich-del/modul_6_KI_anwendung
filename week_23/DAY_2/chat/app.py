import streamlit as st
import ollama

st.set_page_config(page_title="Ollama Chat", page_icon="💬")

st.title("💬 Ollama Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
prompt = st.chat_input("Type your message...")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Send full conversation to Ollama
    response = ollama.chat(
        model="llama3.2",
        messages=st.session_state.messages
    )

    reply = response["message"]["content"]

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(reply)