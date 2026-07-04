from langchain_community.vectorstores import Chroma
from llm.factory import ModelFactory
from config.settings import VECTOR_DB_DIR

def create_vectorstore(docs):

    if not docs:
        print("⚠️ keine docs")
        return None

    return Chroma.from_documents(
        documents=docs,
        embedding=ModelFactory.embeddings(),
        persist_directory=str(VECTOR_DB_DIR)
    )


def load_vectorstore():

    return Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=ModelFactory.embeddings()
    )