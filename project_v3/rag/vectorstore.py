import os
from langchain_chroma import Chroma
from llm.client import get_embeddings
from config.settings import VECTOR_DB_DIR


def create_vectorstore(documents):
    if not documents:
        print("⚠️ keine docs")
        return None

    embeddings = get_embeddings()

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )


def load_vectorstore():
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
    )