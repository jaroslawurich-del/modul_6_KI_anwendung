from pathlib import Path

import streamlit as st

from langchain_core.documents import Document

from config.settings import UPLOAD_DIR

from rag.indexing import rebuild_index


def upload_ui():

    st.subheader("Dokumente")

    uploaded_files = st.file_uploader(
    "Dateien auswählen",
    accept_multiple_files=True,
    type=[
        "txt",
        "md",
        "csv",
        "pdf"
    ]
)

    if not uploaded_files:
        return

    Path(UPLOAD_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    docs = []

    for file in uploaded_files:

        data = file.read()

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode(
                "latin-1",
                errors="ignore"
            )

        path = Path(UPLOAD_DIR) / file.name

        path.write_text(
            text,
            encoding="utf-8"
        )

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": file.name
                }
            )
        )

    rebuild_index(docs)

    st.success(
        f"{len(docs)} Dokument(e) indexiert."
    )