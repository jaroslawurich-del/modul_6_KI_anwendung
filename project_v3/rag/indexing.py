from rag.splitter import split_documents
from rag.vectorstore import create_vectorstore


def rebuild_index(documents):
    """
    Baut den Vektorindex neu auf (sicher für Chroma + Docker + Windows).
    """

    if not documents:
        print("⚠️ keine docs")
        return None

    # Dokumente splitten
    chunks = split_documents(documents)

    if not chunks:
        print("⚠️ keine chunks nach splitting")
        return None

    # Chroma erstellt / updated die DB selbst sicher
    vectorstore = create_vectorstore(chunks)

    return vectorstore