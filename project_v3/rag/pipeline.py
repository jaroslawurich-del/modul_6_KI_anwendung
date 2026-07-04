from rag.retriever import get_retriever


def search(question: str):

    retriever = get_retriever()

    docs = retriever.invoke(question)

    return docs


def build_context(question: str):

    docs = search(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context