from rag.vectorstore import load_vectorstore
from config.settings import TOP_K

def get_retriever():

    db = load_vectorstore()

    return db.as_retriever(
        search_kwargs={"k": TOP_K}
    )