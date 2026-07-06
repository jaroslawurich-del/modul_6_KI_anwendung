from pathlib import Path
import streamlit as st

from config.settings import UPLOAD_DIR
from rag.loader import load_file
from rag.indexing import rebuild_index


def upload_ui():

    st.subheader("Dokumente 📄")

    uploaded_files = st.file_uploader(
        "Dateien auswählen",
        accept_multiple_files=True,
        type=["txt", "md", "csv", "pdf"]
    )

    if not uploaded_files:
        return

    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    all_docs = []

    for file in uploaded_files:

        file_path = Path(UPLOAD_DIR) / file.name

        # Save binary-safe (WICHTIG für PDFs)
        with open(file_path, "wb") as f:
            f.write(file.read())

        try:
            docs = load_file(str(file_path))
            all_docs.extend(docs)

        except Exception as e:
            st.error(f"Fehler bei {file.name}: {e}")

    if all_docs:
        rebuild_index(all_docs)

        st.success(f"{len(all_docs)} Dokumente (inkl. PDFs) indexiert.")