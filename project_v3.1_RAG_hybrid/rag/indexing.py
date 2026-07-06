from rag.splitter import split_documents
from rag.vectorstore import create_vectorstore
from utils.hashing import hash_text


def rebuild_index(documents):

    if not documents:
        return None

    chunks = split_documents(documents)

    if not chunks:
        return None

    for c in chunks:
        c.metadata = c.metadata or {}
        c.metadata["chunk_hash"] = hash_text(c.page_content)

    return create_vectorstore(chunks)