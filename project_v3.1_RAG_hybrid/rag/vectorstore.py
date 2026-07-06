import os
from langchain_chroma import Chroma
from llm.client import get_embeddings

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "/data/chroma_db")


def get_vectorstore():
    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=get_embeddings(),
    )


def create_vectorstore(chunks):
    vectorstore = get_vectorstore()

    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]
    ids = [c.metadata.get("chunk_hash") for c in chunks]

    vectorstore.add_texts(
        texts=texts,
        metadatas=metas,
        ids=ids
    )

    return vectorstore