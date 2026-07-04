from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from config.settings import DOCUMENT_DIR

def load_documents():

    docs = []

    for pdf in Path(DOCUMENT_DIR).glob("*.pdf"):

        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        for p in pages:
            p.metadata["source"] = pdf.name

        docs.extend(pages)

    return docs