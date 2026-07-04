import streamlit as st
from config.settings import DOCUMENT_DIR
from rag.indexing import rebuild_index

def upload():

    file = st.file_uploader("PDF", type="pdf")

    if file:

        path = DOCUMENT_DIR / file.name

        with open(path, "wb") as f:
            f.write(file.getbuffer())

        st.success("hochgeladen")

        rebuild_index()