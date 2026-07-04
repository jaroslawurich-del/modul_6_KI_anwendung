from rag.vectorstore import load_vectorstore

from config.settings import MAX_RESULTS


def get_retriever():

    db = load_vectorstore()

    return db.as_retriever(
        search_kwargs={
            "k": MAX_RESULTS
        }
    )